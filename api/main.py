from typing import List, Dict, Union
from fastapi import FastAPI, Body
from pathlib import Path
import re
import pickle

app = FastAPI()

# Constants
ROOT_DIRECTORY = Path(__file__).parent.parent
MODEL_PATH = ROOT_DIRECTORY / 'artifacts' / 'model'/'best_model.pkl'

# Load the model
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)


def text_preprocessing(products: List[Dict[str, Union[str, int]]]) -> List[str]:
    """
    Preprocesses the list of products to extract the text data.

    Args:
    products (List[Dict[str, Union[str, int]]]): List of product dictionaries.

    Returns:
    List[str]: List of processed text data.
    """
    return [re.sub(r"[0-9]+", "", f"{product['name']};{product['description']}") for product in products]

# def preprocess_text(text: str) -> str:
#     """
#     Processes a given text by removing numbers and applying other preprocessing steps.
#
#     Args:
#     text (str): Text to preprocess.
#
#     Returns:
#     str: Processed text.
#     """
#     return re.sub(r"[0-9]+", "", text)
#
# def text_preprocessing(products: List[Dict[str, Union[str, int]]]) -> List[str]:
#     """
#     Preprocesses the list of products to extract the text data.
#
#     Args:
#     products (List[Dict[str, Union[str, int]]]): List of product dictionaries.
#
#     Returns:
#     List[str]: List of processed text data.
#     """
#     return [preprocess_text(product["name"] + ';' + product["description"]) for product in products]

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/infer")
def infer(products: List[Dict[str, Union[str, int]]] = Body(...)):
    x_test = text_preprocessing(products)
    pred_test = model.predict(x_test)
    return [{'pattern': pattern, 'product_id': product["product_id"]} for pattern, product in zip(pred_test, products)]

