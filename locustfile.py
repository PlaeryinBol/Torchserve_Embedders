from locust import HttpUser, task, between

API_URL = "http://127.0.0.1:9980/predictions/text_embedder"
DUMMY_SAMPLE_PATH = "./sample_text.txt"


class QuickstartUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def on_start(self) -> None:
        self.client.post(API_URL, json=open(DUMMY_SAMPLE_PATH, 'rb').read())
