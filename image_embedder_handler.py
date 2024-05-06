import base64
import io
import json
import logging
import os
import time

import numpy as np
import sentence_transformers as st
import torch
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("sentence_transformers version %s", st.__version__)


class Handler(BaseHandler):
    """This handler takes a image (or image list) and encoding it with SentenceTransformer."""
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.initialized = False

    def initialize(self, ctx) -> None:
        """Loads the model.pt file and initialized the model object."""
        if self.debug:
            properties = {"gpu_id": 0, "model_dir": '.'}
        else:
            self.manifest = ctx.manifest
            properties = ctx.system_properties
        logger.info(properties)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        setup_config_path = os.path.join(properties.get("model_dir"), "models_info.json")
        # loading setup_config_file from disk
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        self.image_encoder = st.SentenceTransformer(self.setup_config['image_embedder_name'],
                                                    device=self.device,
                                                    cache_folder=self.setup_config['image_embedder_dir'])
        self.image_encoder.eval()
        self.initialized = True

    def preprocess(self, request) -> list:
        """The input request data is passed to preprocess. Here the request is parsed and returned as list of images."""
        logger.info(f'input batch size: {len(request)}')
        image_list = []
        for _, data in enumerate(request):
            image = data.get("data") or data.get("body")
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = Image.open(io.BytesIO(image))
            image_list.append(image)
        return image_list

    def inference(self, images: list) -> np.array:
        """ The preprocessed images is passed to inference to obtain encoded features with SentenceTransformer."""
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.time()

        with torch.no_grad():
            image_features = self.image_encoder.encode(images, show_progress_bar=False)

        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
        else:
            end = time.time()

        if torch.cuda.is_available():
            duration = start.elapsed_time(end)
        else:
            duration = (end - start) * 1000

        logger.info(f'duration: {duration} ms')
        return image_features

    def postprocess(self, inference_output: np.array) -> list:
        """Postprocess function converts ndarrays to list."""
        logger.info(f'predict shape: {inference_output.shape}')
        return inference_output.tolist()

    def handle(self, data, context) -> list:
        """Entry point for data preprocessing, inference and post processing."""
        if not self.initialized:
            self.initialize(context)

        if data is None:
            return None

        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)
        return data


if __name__ == '__main__':
    with open('./sample_image.jpg', 'rb') as f:
        file_string = f.read()
    dummy_request = [{'data': file_string}]
    service = Handler(debug=True)
    result = service.handle(dummy_request, None)
    print(f'OK, output shape: {len(result[0])}')
