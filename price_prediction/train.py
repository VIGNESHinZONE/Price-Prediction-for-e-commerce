from os.path import join
import pandas as pd
import gzip
from price_prediction import NaiveFeaturiser
from price_prediction import PricePredictionModel
from price_prediction import build_crossvalidation_data

data_path = join(*["data"])
save_path = join(*["weights"])

if __name__ == '__main__':
    with gzip.open(join(data_path, "mercari_train.csv.gz")) as f:
        df_train = pd.read_csv(f)
    with gzip.open(join(data_path, "mercari_test.csv.gz")) as f:
        df_test = pd.read_csv(f)

    ids = df_test['id'].to_numpy()
    featuriser, df_train, df_test = NaiveFeaturiser \
        .build_from_dataframe(df_train, df_test)

    # Saving the Featuriser weights
    featuriser.save(join(save_path, "featuriser.pkl"))

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
        'alpha': 0.4,
        'max_iter': 200,
        'tol': 0.01,
    }

    lgbm_args_1 = {
        'learning_rate': 0.85,
        'max_depth': 3,
        'num_leaves': 110,
        'verbosity': -1,
        'metric': 'RMSE',
    }

    lgbm_args_2 = {
        'learning_rate': 0.76,
        'max_depth': 3,
        'num_leaves': 99,
        'verbosity': -1,
        'metric': 'RMSE',
    }
    train_datasets, val_datasets = \
        build_crossvalidation_data(df_train, split=3)

    for i, dataset in enumerate(zip(train_datasets, val_datasets)):
        train, val = dataset
        print(f"Fold {i} \n")
        model = PricePredictionModel(name_args, category_args, brand_args,
                                     item_args, ridge_model_args, lgbm_args_1,
                                     lgbm_args_2)
        model.fit(train, val)

    # model = PricePredictionModel(name_args, category_args, brand_args,
    #                              item_args, ridge_model_args, lgbm_args_1,
    #                              lgbm_args_2)
    # model.fit(df_train)
    # preds = model.predict_df(df_test)

    # # Saving model weights
    # model.save(join(save_path, "model_weights.pkl"))
    # submit_df = pd.DataFrame({"id": ids, "price": preds})
    # submit_df.to_csv("submission.csv", index=False)
