# Torchserve Embedders

## Overview
The repository contains code for running two models ([text](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) and [image](https://huggingface.co/sentence-transformers/clip-ViT-B-32) embedders) with torchserve.  

## Usage

1. Installing dependencies
```bash
git clone https://github.com/pytorch/serve.git
```

```bash
cd serve
python ./ts_scripts/install_dependencies.py --cuda=cu121
```

```bash
pip install -r requirements.txt
```

2. Download models from HuggingFace

```bash
export HF_HOME=<folder for storing models>
export HF_HUB_CACHE=<folder for storing models>
huggingface-cli download sentence-transformers/paraphrase-multilingual-mpnet-base-v2 sentence-transformers/clip-ViT-B-32
```

3. Check the models, save them to .bin files
```bash
python convert_models_to_bin.py
```

4. Create .mar files from models for serving using handler files
```bash
. scripts/create_mar_files.sh
```

5. Specify the necessary parameters in *config.properties*, start the server
```bash
. scripts/torchserve_start.sh
```

6. Measuring performance using *locustfile.py*
```bash
. scripts/locust_test.sh
```

**Examples of sending requests:**
```bash
curl -X POST http://127.0.0.1:9980/predictions/text_embedder -T ./sample_text.txt
curl -X POST http://127.0.0.1:9980/predictions/image_embedder -T ./sample_image.jpg
```

**Performance Tests**
For two separate models with 1 worker:
* *batchSize=8* - 520 rps
* *batchSize=16* - 550 rps
* *batchSize=32* - 580 rps  

When running both models simultaneously, the best result of 460 rps is achieved with *batchSize=8*.

**TODO**
* try optimizing models via *TensorRT/ONNX*
* before sending the image to the model, resize it to the input size of the model (to reduce the amount of bytes sent)
* optimize data transfer - use *pickle* and *imageio*
* pass [metrics](https://pytorch.org/serve/metrics_api.html) to *Prometheus*
* use *docker/docker-compose*