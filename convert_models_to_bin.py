import os
import shutil

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

SOURCE_MODELS_DIR = '/home/gpuuser/.cache/huggingface/hub'
BIN_MODELS_DIR = './model_store'
TEXT = ['Hello world!', 'Привет, мир!']
IMAGE = './sample_image.jpg'
MODELS_CONFIG = {
    'text': {
        'source': [
            'models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2/snapshots/79f2382ceacceacdf38563d7c5d16b9ff8d725d6'
        ],
        'bin_version': [
            'text_embedder'
        ]
    },
    'vision': {
        'source': [
            'sentence-transformers_clip-ViT-B-32'
        ],
        'bin_version': [
            'image_embedder'
        ]
    }
}


# run models on test examples, save them in binary format
def run_and_convert2bin() -> None:
    os.makedirs(BIN_MODELS_DIR, exist_ok=True)
    for model_type in MODELS_CONFIG:
        for i, model in enumerate(MODELS_CONFIG[model_type]['source']):
            filename = MODELS_CONFIG[model_type]['bin_version'][i]
            m = SentenceTransformer(os.path.join(SOURCE_MODELS_DIR, model))

            with torch.no_grad():
                if model_type == 'text':
                    embeddings = m.encode(TEXT)
                else:
                    embeddings = m.encode(Image.open(IMAGE))

            save_dir = os.path.join(BIN_MODELS_DIR, filename)
            m.save(save_dir)
            shutil.make_archive(save_dir, 'zip', save_dir)
            bin_file = save_dir + '.zip'
            os.rename(bin_file, bin_file.replace('.zip', '.bin'))
            shutil.rmtree(save_dir)
            print(f'{filename} OK, output shape: {embeddings.shape}')


if __name__ == '__main__':
    run_and_convert2bin()
