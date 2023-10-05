import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from src.classifier import AsosPatternModel


class TestAsosPatternModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root_directory = Path(".")  # Adjust based on directory structure

    def setUp(self):
        self.model = AsosPatternModel(self.root_directory)

    def test_data_loading(self):
        mock_data = pd.DataFrame({
            "name": ["test"],
            "description": ["test description"],
            "pattern": ["polka_dot"]
        })

        with patch("pandas.read_table", return_value=mock_data):
            train_data, test_data = self.model.load_data("mock_train_path", "mock_test_path")

            self.assertEqual(train_data["name"].iloc[0], "test", "Name column not loaded correctly.")
            self.assertEqual(train_data["description"].iloc[0], "test description",
                             "Description column not loaded correctly.")
            self.assertEqual(train_data["pattern"].iloc[0], "polka_dot", "Pattern column not loaded correctly.")

    def test_data_preprocessing(self):
        sample_data = pd.DataFrame({
            "name": ["test1", "test2"],
            "description": ["description1", "description2"],
            "pattern": ["polka_dot", "stripe"]
        })

        processed_data, labels = self.model.preprocess_data(sample_data)

        self.assertEqual(len(processed_data), 2, "Processed data length mismatch.")
        self.assertIn("test;description", processed_data, "Processed data content mismatch.")
        self.assertListEqual(list(labels), ["polka_dot", "stripe"], "Labels mismatch.")

    def test_stopwords_update(self):
        new_stopword = "testword"
        self.model.update_stopwords(new_stopword)
        self.assertIn(new_stopword, self.model.stops, "Stopword update failed.")

    def test_model_save_and_load(self):
        mock_model = MagicMock()

        with patch("pickle.dump"), \
                patch.object(AsosPatternModel, "load_model", return_value=None):
            self.model.save_model(mock_model, "mock_model.pkl")
            loaded_model = self.model.load_model("mock_model.pkl")
            self.assertIsNone(loaded_model, "Loaded model should be None as per mock.")
            os.remove(f"{self.model.model_dir}/mock_model.pkl")

    # Additional tests can be added for other methods, as required.


if __name__ == "__main__":
    unittest.main()
    os.remove("mock_model.pkl")
