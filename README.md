
# Pattern Prediction Model 

The production-ready refactored code of prototype code is written at src/classifier.py and for the scope of continuous experimentation developed modular experiment.py, to run exploratory data analysis, feature selection, model experimentation without affecting the pre-written production code.

Saved the model from production ready code and able to load it to accept and process inference Predictions requests through FastAPI deployment, by running the FastAPI/main.py, follow the instructions below. 

### Pattern Predictor 

```
patternpredictor2/
│
├── artifacts/
│   └── model/
│       └── best_model.pkl
│
├── data/
│   ├── exercise_test.csv
│   ├── exercise_train.csv
│   └── predictions_test_text.csv
│
├── api/
│   ├── main.py
│   └── __init__.py
│
├── logs/
│   ├── model_evaluation.log
│   ├── api_test.py
│   └── __init__.py
│
├── src/
│   ├── classifier.py
│   ├── experiment.py
│   └── test_classifier.py
├── tests/
│   ├── test_classifier.py
│   └── api_test.py
│
├──run.sh
├── README.md
├── requirements.txt

```

## Appendix

The ```experiment.py ``` to cater the requirements of data scientists for continuous experiment. And model results for both baseline and optimized are found in model_evaluation.log saved with timestamps of run.

Implementation of CI/CD would be optimal for project.

### Things I missed to add and consider with more time.

Implement MLflow for experiment tracking, dynamic logging and better versioning for the experiments. Hence, better access to past models and data.

Store the inference predictions with data it predicted for future groundtruthing them and evaluating the model performance over new data.

Obtain more labelled new data and test inference predictions, obtain the classification report and other metrics like F1-Score, Macro-average, weighted average, to better understand the predictions. without that I was hesitant to include these currently. However, it would be very ideal.

### Problems I noticed with the current prototype

There is no label for pattern in test data, hence unable to assess the model performance for inference and unable to obtain classification report, for the same. 

The model is as good as its data, hence evaluating solely on the train data is not optimal.

The cross-validation script in good and improved the accuracy better, however it was within the train set itself.

I missed the conda environment part, took sometime to clean my PC space for anaconda install.

### Evaluation of API performance and model inference

I have developed locust API performance stress-test. That will generate all the formal API performance metrics namely Failure,rate, response time, Average size(bytes), etc.

I consider with the time frame and this can be further discussed like Prometheus, scaling with cloud deployment, docker, kubernetes solutions.



## API Reference

#### Get 

```python
@app.get("/")
def home():
    return {"health_check": "OK"}
```
#### Post 
```python
@app.post("/infer")
def infer(products: List[Dict[str, Union[str, int]]] = Body(...)):
    x_test = text_preprocessing(products)
    pred_test = model.predict(x_test)
    return [{'pattern': pattern, 'product_id': product["product_id"]} for pattern, product in zip(pred_test, products)]
```

```http
  POST /infer
```

| Parameter  | Type     | Description                           |
| :--------- | :------- | :------------------------------------ |
| `payload`  | `array`  | Required. JSON array of products. See below   |


#### Product (inside payload list)

| Parameter  | Type     | Description                           |
| :--------- | :------- | :------------------------------------ |
| `name`     | `string` | Required. The name of the product.    |
| `description` | `string` | Required. The product description. |
| `product_id`  | `integer` | Required. The unique identifier for the product. |

| Attribute  | Type     | Description                                  |
| :--------- | :------- | :------------------------------------------- |
| `pattern`  | `string` | The matched pattern in the product description. |
| `product_id` | `integer` | The unique identifier of the product associated with the matched pattern. |



## Deployment


```


 Run Locally
----------------------

### Setup on New Computer:

1.  **Ensure Conda Python Packages is installed.**

2.  **Navigate to the project directory.**

3.  **Create a new virtual environment:**


     `python -m venv asosexercise`
4. However, you can skip 3 and directly run.sh, the model running explicitly requires Python v3.7
* * * * *

```bash
cd patternpredictor #project directory

# Ensure the run.sh script has execute permissions
chmod +x run.sh

# Execute the run.sh script to train the model and start the api server
./run.sh

```
The model is deployed from the saved model, If you wish run the training of model, simply input 'yes' when asked for the same during the run.sh 



## Code Refactored


src
------------

The script ````classify.py ````  is developed to predict product material patterns (e.g. spotty, stripy, colored) from metadata such as product descriptions. The code structure primarily consists of a dedicated class named `AsosPatternModel` that encompasses methods for data loading, preprocessing, training, hyperparameter tuning, as well as saving and reloading of the model. Moreover, auxiliary functions for logging and the primary script execution are integrated. 

I was considering more modular deployment for creating each method with SOLID, however given the model to be production-ready and codebase, I opted this style with adding the ```experiment.py``` to cater the need for continuos experimenation without affecting the production code.

Features
--------

-   **Logging**: Metrics related to model training such as accuracy, classification report, and confusion matrix are systematically recorded into `model_evaluation.log`.

-   **Data Loading & Preprocessing**: Integrated methods facilitate the loading of tab-delimited data. Data preprocessing includes removal of numbers and amalgamation of the 'name' and 'description' fields.

-   **Training**: The training phase utilizes a pipeline composed of `TfidfVectorizer` and `SGDClassifier` to establish a baseline model.

-   **Hyperparameter Tuning**: The script employs `GridSearchCV` to fine-tune the hyperparameters of the model, ensuring optimal performance.

-   **Model Persistence**: Functionality for saving and reloading the trained model is embedded, allowing for future model deployments and evaluations without the necessity for retraining.
## Running Tests

To run tests, run the following command

1.  Execute the `tests/` script:

bash

`cd tests`

Follow the prompts to choose which test to run: 
1. `test_classifier.py` or 
2. `api_test.py`.

