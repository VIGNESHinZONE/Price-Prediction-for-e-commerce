import os
import tempfile
import numpy as np
import pandas as pd
from price_prediction import NaiveFeaturiser
from price_prediction import PricePredictionModel


def test_save_load_model():
    test_csv = pd.read_csv(
        os.path.join(*[os.getcwd(), "tests", "sample_train.csv"]))
    _, test_csv, _ = NaiveFeaturiser.build_from_dataframe(test_csv)

    name_args = {
        'max_features': 1000,
        'min_df': 10,
        'stop_words': 'english',
    }
    category_args = {}
    brand_args = {'sparse_output': True}
    item_args = {
        'max_features': 500,
        'ngram_range': (1, 3),
        'stop_words': 'english'
    }
    ridge_model_args = {
        'solver': "auto",
        'fit_intercept': True,
        'random_state': 205
    }
    lgbm_args_1 = {
        'learning_rate': 0.75,
        'max_depth': 2,
        'num_leaves': 50,
        'verbosity': -1,
        'metric': 'RMSE',
        'verbose_eval': 100
    }
    lgbm_args_2 = {
        'learning_rate': 0.76,
        'max_depth': 3,
        'num_leaves': 99,
        'verbosity': -1,
        'metric': 'RMSE',
    }

    model = PricePredictionModel(name_args, category_args, brand_args,
                                 item_args, ridge_model_args, lgbm_args_1,
                                 lgbm_args_2)
    model.fit(test_csv, test_csv)
    preds = model.predict_df(test_csv)
    tmp_file = tempfile.NamedTemporaryFile()
    model.save(tmp_file.name)
    tmp_file.flush()
    new_model = PricePredictionModel.load(tmp_file.name)
    new_preds = new_model.predict_df(test_csv)
    assert np.allclose(preds, new_preds)
