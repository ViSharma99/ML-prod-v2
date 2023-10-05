import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import numpy as np
from classifier import AsosPatternModel


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperimentClass:

    @staticmethod
    def split_train(df, test_size=0.1, random_state=42, stratify_column='pattern'):
        """
        Split the train dataset into train and validation sets.

        Args:
        df (pd.DataFrame): Dataframe to split.
        test_size (float): Ratio of trainset to use for validation.
        random_state (int): Specify the randomness (entropy) of splitting.
        stratify_column (str): Column or attributes to which splitting must be proportional.

        Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes.
        """
        stratify = df[stratify_column].values
        ds_train, ds_validate = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
        logger.info('Split_trainset: Success')
        return ds_train, ds_validate

    @staticmethod
    def feature_eng(model, df_train, target_column='pattern'):
        """
        To Run further data analysis for the data scientists, find any novel patterns and figure change to target columns.

        Args:
        model (Pipeline): Trained model for extracting features.
        df_train (pd.DataFrame): Dataframe for analysis.
        target_column (str): Target column for analysis.
        """
        ctab = pd.crosstab(index=df_train[target_column], columns=target_column)

        # Extract model coefficients and feature names (token) from the pipeline
        coefs = model.named_steps['clf'].coef_
        tvec = np.array(model.named_steps['tfidf_vect'].get_feature_names())

        # Check the model coefficient to get the most "important" features per class
        for i in range(coefs.shape[0]):
            top5 = np.argsort(-coefs[i])[:5]
            print(ctab.index[i], tvec[top5])

        # Check model sparsity
        for i in range(coefs.shape[0]):
            print(ctab.index[i], "sparsity", np.sum(coefs[i] == 0) / coefs.shape[1])


if __name__ == "__main__":
    ROOT_DIRECTORY = Path(__file__).parent.parent
    train_data = ROOT_DIRECTORY / "data" / "exercise_train.tsv"
    test_data = ROOT_DIRECTORY / "data" / "exercise_test.tsv"

    asos_model = AsosPatternModel(ROOT_DIRECTORY)
    df_train, df_test = asos_model.load_data(train_data, test_data)
    ds_train, ds_validate = ExperimentClass.split_train(df=df_train)

    asos_model.update_stopwords('dress', 'model', 'wears', 'fit', 'true', 'to', 'size', 'uk', 'us', 'tall', 'cm')
    x_train, y_train = asos_model.preprocess_data(ds_train)
    base_model = asos_model.train_baseline(x_train, y_train)

    ExperimentClass.feature_eng(base_model, df_train)
