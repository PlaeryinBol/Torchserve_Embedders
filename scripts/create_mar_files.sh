torch-model-archiver --model-name text_embedder \
    --version 1.0 \
    --serialized-file ./model_store/text_embedder.bin \
    --export-path ./model_store \
    --handler ./text_embedder_handler.py \
    --extra-files ./models_info.json \
    --force
torch-model-archiver --model-name image_embedder \
    --version 1.0 \
    --serialized-file ./model_store/image_embedder.bin \
    --export-path ./model_store \
    --handler ./image_embedder_handler.py \
    --extra-files ./models_info.json \
    --force