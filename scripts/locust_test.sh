locust --headless \
    --users 1000 \
    --spawn-rate 1000 \
    --run-time 60 \
    --host http://127.0.0.1:9980/predictions/text_embedder
