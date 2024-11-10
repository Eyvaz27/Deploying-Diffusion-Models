import gc
import os

import yaml
import torch
import numpy as np
from PIL import Image
from torch import Tensor, nn
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from pipeline import CustomDiffusionPipeline, CustomDiffusionPipelineCfg

CONFIG_PATH="config.yaml"

def load_config():
    with open(CONFIG_PATH, 'r') as file:
        configs = yaml.safe_load(file)
    cfg = CustomDiffusionPipelineCfg(**configs)
    return configs, cfg

def encode_image(self, image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.read())

class PyTritonServer:

    def __init__(self):
        """
        Initialize the PyTritonServer class and load the model.
        """
        # NOTE: you can set your model name here
        configs, cfg = load_config()
        self.pipeline = CustomDiffusionPipeline(cfg)

        # NOTE: you can change the device to "cpu" if you don't have a GPU, otherwise you may perform any optimizations you may find necessary
        self.device = torch.device(configs["device"])

        if os.getenv("WARMUP_MODEL", False):
            self._warmup()
        

    def _warmup(self):
        """
        Why would you want to warm up your model?
        """
        # TODO: implement model warmup
        # TODO: explain why this might be necessary
        # Truthfully speaking I didn't get what do you mean by WARM-UP
        # 1) as long as we are running models in inference mode we don't need any WARM-UP (of learning rate)
        # 2) if you imply that checkpoints should be uploaded first during initialization, that is a usual practice ->
        #   -> you wouldn't like to load checkpoints at every query right?
        self.pipeline.init_pipeline()
        self.pipeline.device_load()

    def _infer_fn(self, requests):
        """
        The inference function that will be called by Triton when a request is made.
        
        This processes a list of requests and returns a list of responses, but in this case, we only have one request at a time.
        """
        responses = []
        for req in requests:
            req_data = req.data["requests"]
            # TODO: interpret the request data
            prompt = np.char.decode(req_data.astype("bytes"), "utf-8").item()
            # TODO: implement model inference
            image = self.pipeline.generate(prompt)
            # TODO: transform the model output into a byte array 
            byte_image = encode_image(image)
            responses.append({"result": np.array([byte_image])})

        # NOTE: this is important to free up memory
        gc.collect()
        torch.cuda.empty_cache()
        return responses

    def run(self):
        """
        The main function that runs the Triton server and sets up the bindings.
        """
        with Triton(
            config=TritonConfig(
                # TODO: set up the triton server configuration
                exit_on_error=True, log_verbose=1)) as triton:
            triton.bind(
                model_name="CustomDiffusionPipeline",  # TODO: set up the model name
                infer_func=self._infer_fn,
                inputs=[
                    # TODO: set up model inputs
                    Tensor(name="requests", dtype=np.bytes_, shape=(1,)),
                ],
                outputs=[
                    # TODO: set up model outputs
                    Tensor(name="result", dtype=np.bytes_, shape=(1,)),
                    ],
                config=ModelConfig(batching=False),
            )
            triton.serve()

if __name__ == "__main__":
    server = PyTritonServer()
    server.run()
