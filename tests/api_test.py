from locust import HttpUser, task, between
import pandas as pd
import os
import random  # Import the random module


class FastAPIUser(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):  # This function runs when a simulated user starts
        directory = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
        df_test = pd.read_table(f"{directory}/data/exercise_test.tsv")
        df_test_selected = df_test[["name", "description", "productId"]]
        df_test_selected.columns = ['name', 'description', 'product_id']
        self.test_data = df_test_selected.to_dict(orient='records')  # Assign to an instance variable to access in tasks

    @task
    def predict_endpoint(self):
        headers = {"Content-Type": "application/json"}

        # Randomly select a row from test_data
        random_row = random.choice(self.test_data)

        # Use the randomly selected row as the payload
        self.client.post("/infer", json=[random_row], headers=headers)  # Ensure it's a list


if __name__ == "__main__":
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    os.system(f"locust -f {directory}/api_test.py --host=http://127.0.0.1:8000")
