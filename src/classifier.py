import re
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from pathlib import Path
import logging
import pickle


def configure_logging():
    logging.basicConfig(filename='logs/model_evaluation.log', level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


logger = configure_logging()


class AsosPatternModel:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.stops = set(text.ENGLISH_STOP_WORDS)
        self.model_dir = root_dir / "artifacts/model"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, train_path, test_path):
        """
        Loads the training and testing datasets.

        Args:
        train_file (Path): Path for the training dataset.
        test_file (Path): Path for the testing dataset.

        Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing dataframes for training and testing data.
        """
        return pd.read_table(self.root_dir / train_path), pd.read_table(self.root_dir / test_path)

    def preprocess_data(self, df, target_column='description', sep=';'):
        """
        Processes the input dataframe by concatenating columns and removing numbers.

        Args:
        df (pd.DataFrame): Dataframe object to preprocess.
        target_column (str, optional): Feature to preprocess the text, more likely for training model. Defaults to 'description'.
        sep (str, optional): Character to use for concatenating columns. Defaults to ';'.

        Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing processed data and labels.
        """
        df_p = df['name'].str.cat(df[target_column], sep=sep).values
        df_p = [re.sub(r"[0-9]+", "", desc) for desc in df_p]
        return df_p, df.get('pattern', None)

    def update_stopwords(self, *words):
        """
        Updates the set of stopwords.

        Args:
        *words (str): Words to be added to the stopwords set.
        """
        self.stops.update(words)

    def train_baseline(self, x_train, y_train, **kwargs):
        """
        Trains the baseline model.

        Args:
        x_train (np.ndarray): The training data.
        y_train (np.ndarray): The training labels.
        **kwargs: Additional keyword arguments for the SGDClassifier.

        Returns:
        Pipeline: The trained model.
        """
        text_clf = Pipeline([
            ('tfidf_vect', TfidfVectorizer(stop_words=self.stops)),
            ('clf', SGDClassifier(**kwargs))
        ])
        text_clf.fit(x_train, y_train)
        y_pred_train= text_clf.predict(x_train)
        logger.info("Baseline Training Classification Report:\n" + classification_report(y_train, y_pred_train))
        logger.info("Baseline Training Confusion Matrix:\n" + str(confusion_matrix(y_train, y_pred_train)))
        logger.info("Baseline Training Accuracy: " + str(accuracy_score(y_train, y_pred_train)))

        return text_clf

    def tune_hyperparameters(self, model, x_train, y_train, parameters):
        """
        Tunes hyperparameters using GridSearchCV.

        Args:
        model (Pipeline): The baseline model.
        x_train (np.ndarray): The training data.
        y_train (np.ndarray): The training labels.
        parameters (dict): Dictionary of parameters for tuning.

        Returns:
        Pipeline: The best model after hyperparameter tuning.
        """
        skf = StratifiedKFold(n_splits=8)
        grid_search = GridSearchCV(model, param_grid=parameters, n_jobs=-1, cv=skf, verbose=1)
        grid_search.fit(x_train, y_train)
        y_pred_train = grid_search.predict(x_train)
        # Logging metrics
        logger.info("Training Classification Report:\n" + classification_report(y_train, y_pred_train))
        logger.info("Training Confusion Matrix:\n" + str(confusion_matrix(y_train, y_pred_train)))
        logger.info("Training Accuracy: " + str(accuracy_score(y_train, y_pred_train)))

        # Additional metrics can be logged similarly

        return grid_search.best_estimator_

    def save_model(self, model, filename):
        """
        Saves the trained model to a specified path.

        Args:
        model (Pipeline): The trained model to save.
        path (Path): The path where the model should be saved.
        """
        with (self.model_dir / filename).open('wb') as model_file:
            pickle.dump(model, model_file)
        logger.info(f'Model saved at {self.model_dir / filename}')

    def load_model(self, filename):
        """
        Loads a saved model from a specified path.

        Args:
        path (Path): The path where the model is saved.

        Returns:
        Pipeline: The loaded model.
        """
        with (self.model_dir / filename).open('rb') as saved_model:
            return pickle.load(saved_model)


def main():
    root_directory = Path(__file__).parent.parent

    model = AsosPatternModel(root_directory)

    df_train, df_test = model.load_data("data/exercise_train.tsv", "data/exercise_test.tsv")

    model.update_stopwords('dress', 'model', 'wears', 'fit', 'true', 'to', 'size', 'uk', 'us', 'tall', 'cm')

    x_train, y_train = model.preprocess_data(df_train)
    x_test, _ = model.preprocess_data(df_test)
    base_model = model.train_baseline(x_train, y_train, loss='modified_huber', penalty='elasticnet', alpha=1e-3, random_state=42)
    hyperparameters = {
        'tfidf_vect__max_df': (0.5, 0.75, 1.0),
        'tfidf_vect__ngram_range': ((1, 1), (1, 2)),
        'clf__alpha': (1e-05, 1e-04, 1e-3),
        'clf__l1_ratio': (0.01, 0.05, 0.1)
    }

    best_model = model.tune_hyperparameters(base_model, x_train, y_train, hyperparameters)
    model.save_model(best_model, "best_model.pkl")


if __name__ == "__main__":
    main()
