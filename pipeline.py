from PIL import Image
import io
import torch
import base64
from torch import Tensor, nn
from dataclasses import dataclass
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from optimum.quanto import freeze, qfloat8, qint4, qint8, quantize
from diffusers import StableDiffusionLatentUpscalePipeline
from diffusers import UniPCMultistepScheduler

UNET_QTYPES = {
    "fp8": qfloat8,
    "int8": qint8,
    "int4": qint4,
    "none": None,
}

@dataclass
class CustomDiffusionPipelineCfg:
  device: str
  height: str
  width: str
  num_inference_steps: int
  guidance_scale: float
  stable_diff_version: str
  quantize: bool
  unet_qtype: str

class CustomDiffusionPipeline(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.device = torch.device(self.cfg.device)

    self.init_pipeline()
    self.device_load()

  def init_pipeline(self):
    self.vae = AutoencoderKL.from_pretrained(self.cfg.stable_diff_version, subfolder="vae", use_safetensors=True, torch_dtype=torch.float16)
    self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.stable_diff_version, subfolder="tokenizer", torch_dtype=torch.float16)
    self.text_encoder = CLIPTextModel.from_pretrained(self.cfg.stable_diff_version, subfolder="text_encoder", use_safetensors=True, torch_dtype=torch.float16)
    self.unet = UNet2DConditionModel.from_pretrained(self.cfg.stable_diff_version, subfolder="unet", use_safetensors=True, torch_dtype=torch.float16)
    self.scheduler = UniPCMultistepScheduler.from_pretrained(self.cfg.stable_diff_version, subfolder="scheduler", torch_dtype=torch.float16)

    # # # UpScaler
    upscaler_model_id = "stabilityai/sd-x2-latent-upscaler"
    self.upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(upscaler_model_id, torch_dtype=torch.float16)

    # # # Applying Model Quantization
    if self.cfg.quantize and self.cfg.unet_qtype:
        quantize(self.unet, weights=UNET_QTYPES[self.cfg.unet_qtype])
        quantize(self.upscaler.unet, weights=UNET_QTYPES[self.cfg.unet_qtype])

  def device_load(self):
    self.vae.to(self.device)
    self.text_encoder.to(self.device)
    self.unet.to(self.device)
    self.upscaler = self.upscaler.to(self.device)

  def init_random_noise(self, prompt):
    batch_size = len(prompt)
    generator = torch.Generator(device=self.device)  # Seed generator to create the initial latent noise
    latents = torch.randn((batch_size, self.unet.config.in_channels, self.cfg.height // 8, 
                           self.cfg.width // 8), generator=generator, device=self.device)
    latents = latents * self.scheduler.init_noise_sigma
    return latents.to(dtype=torch.float16)

  @torch.no_grad()
  def embed_text_prompt(self, prompt):
    batch_size = len(prompt)
    text_prompt_tokenized = self.tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt",
                                           max_length=self.tokenizer.model_max_length)
    text_embeddings = self.text_encoder(text_prompt_tokenized.input_ids.to(self.device))[0]

    max_length = text_prompt_tokenized.input_ids.shape[-1]
    uncond_input = self.tokenizer([""] * batch_size, padding="max_length", 
                                  max_length=max_length, return_tensors="pt")
    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings

  @torch.no_grad()
  def denoise(self, prompt):

    # initialize latents and embed text first
    latent_code = self.init_random_noise(prompt)
    text_embeddings = self.embed_text_prompt(prompt)
    self.scheduler.set_timesteps(self.cfg.num_inference_steps)
    
    # iterate through denoising steps
    for t in self.scheduler.timesteps:
      latent_model_input = torch.cat([latent_code] * 2)
      latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
      # predict the noise residual
      noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
      # perform guidance
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
      # compute the previous noisy sample x_t -> x_t-1
      latent_code = self.scheduler.step(noise_pred, t, latent_code).prev_sample
    # # return latents after denoising
    return latent_code

  @torch.no_grad()
  def generate(self, prompt):
    latents = 1 / 0.18215 * self.denoise(prompt)
    # decode the image latents with vae
    image = self.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    return image
  
  @torch.no_grad()
  def generate_up(self, prompt):
    low_resolution_image = self.generate(prompt)
    return self.upscaler(prompt=prompt, image=low_resolution_image, 
                         num_inference_steps=20, guidance_scale=0).images[0]