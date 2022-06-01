import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor


class PricePredictionModel(object):
    """
    The Model class for building the Price Prediction task,
    where we build the model from the training dataframe provided
    to us. This model performs the following model operations

        [1] Building a bag of words model for the ["name", "category_name"]
            features using the CountVectorizer.
        [2] One hot encoding for all the major ["brand_name"]. We have already
            preprocessed it either consist of popular brands or "missing".
        [3] Tfidf transformation for the ["item_description"] feature.
        [4] We build two models using the above features
            * Ridge Model
            * LGBM Model
            And the scores from both the models are aggregated.

    This class has been built similar to the Scikit Learn API.
    """
    def __init__(self,
                 name_f,
                 category_f,
                 brand_name_f,
                 item_descp_f,
                 ridge_model,
                 lgbm_model_1,
                 lgbm_model_2,
                 fit_performed=False):
        """
        Parameters
        ----------
        name_f: Union[dict, CountVectorizer]
            arguments for processing the ["name"] features.
        category_f: Union[dict, CountVectorizer]
            arguments for processing the ["category_name"] features.
        brand_name_f: Union[dict, LabelBinarizer]
            arguments for processing the ["brand_name"] features.
        item_descp_f: Union[dict, TfidfVectorizer]
            arguments for processing the ["item_description"] features.
        ridge_model: Union[dict, Ridge]
            the arguments for the Ridge model.
        lgbm_model: Union[dict, LGBMRegressor]
            the arguments for the LGBM model.
        """
        if isinstance(name_f, dict):
            self.feat_name = \
              CountVectorizer(**name_f)
        else:
            self.feat_name = name_f

        if isinstance(category_f, dict):
            self.feat_category = \
              CountVectorizer(**category_f)
        else:
            self.feat_category = category_f

        if isinstance(brand_name_f, dict):
            self.feat_brand = \
              LabelBinarizer(**brand_name_f)
        else:
            self.feat_brand = brand_name_f

        if isinstance(item_descp_f, dict):
            self.feat_item_descp = \
              TfidfVectorizer(**item_descp_f)
        else:
            self.feat_item_descp = item_descp_f

        if isinstance(ridge_model, dict):
            self.ridge_model = \
              Ridge(**ridge_model)
        else:
            self.ridge_model = ridge_model

        if isinstance(lgbm_model_1, dict):
            self.lgbm_model_1 = \
              LGBMRegressor(**lgbm_model_1)
        else:
            self.lgbm_model_1 = lgbm_model_1

        if isinstance(lgbm_model_2, dict):
            self.lgbm_model_2 = \
              LGBMRegressor(**lgbm_model_2)
        else:
            self.lgbm_model_2 = lgbm_model_2

        self.fit_performed = fit_performed

    def predict(self, data: dict):
        """
        Predicts the output to a single entry of request

        Parameters
        ----------
        data: dict
            The input data to be predicted on.

        Returns
        -------
        pred: int
            The predicted price.
        """
        if self.fit_performed is False:
            raise AssertionError("Build the model with `fit` method and "
                                 "then use `predict` method.")

        x_name = self.feat_name.transform([data["name"]])
        x_cat = self.feat_category.transform([data["category_name"]])
        x_brand = self.feat_brand.transform([data["brand_name"]])
        x_item = self.feat_item_descp.transform([data["item_description"]])
        x_ict = data["item_condition_id"]
        x_ship = data["shipping"]
        x = hstack((x_ict, x_ship, x_item, x_brand, x_cat, x_name)).tocsr()
        pred = 0.33 * self.lgbm_model_1.predict(x)
        pred += 0.34 * self.lgbm_model_2.predict(x)
        pred += 0.33 * self.ridge_model.predict(x)
        pred = np.expm1(pred)
        return pred.item()

    def fit(self, df_train: pd.DataFrame, df_valid: pd.DataFrame = None):
        """
        Used to build the models.

        Parameters
        ----------
        df_train: pd.DataFrame
            The training data to build the model.
        df_valid: pd.DataFrame
            The validation data to infer the model performance. Default on None
        """
        Y_train = np.log1p(df_train["price"])
        X_name_train = self.feat_name.fit_transform(df_train['name'])
        X_category_train = \
            self.feat_category.fit_transform(df_train['category_name'])
        X_brand_train = \
            self.feat_brand.fit_transform(df_train['brand_name'])
        X_item_train = \
            self.feat_item_descp.fit_transform(df_train['item_description'])
        X_item_condition_train = csr_matrix(
            df_train.loc[:,
                         'item_condition_id_1':'item_condition_id_5'].values)
        X_shipping_train = csr_matrix(
            df_train.loc[:, 'shipping_0':'shipping_1'].values)
        X_train = hstack(
            (X_item_condition_train, X_shipping_train, X_item_train,
             X_brand_train, X_category_train, X_name_train)).tocsr()

        self.lgbm_model_1 = self.lgbm_model_1.fit(X_train, Y_train)
        pred_train = 0.33 * self.lgbm_model_1.predict(X_train)

        self.lgbm_model_2 = self.lgbm_model_2.fit(X_train, Y_train)
        pred_train += 0.34 * self.lgbm_model_2.predict(X_train)

        self.ridge_model.fit(X_train, Y_train)
        pred_train += 0.33 * self.ridge_model.predict(X_train)
        train_error = mean_squared_error(Y_train, pred_train, squared=False)
        print(f"Train Error - {train_error:5.3f}")
        if df_valid is not None:
            Y_val = np.log1p(df_valid["price"])
            X_name_val = self.feat_name.transform(df_valid['name'])
            X_category_val = \
                self.feat_category.transform(df_valid['category_name'])
            X_brand_val = \
                self.feat_brand.transform(df_valid['brand_name'])
            X_item_val = \
                self.feat_item_descp.transform(df_valid['item_description'])
            X_item_condition_val = csr_matrix(
                df_valid.loc[:, 'item_condition_id_1':'item_condition_id_5'].
                values)
            X_shipping_val = csr_matrix(
                df_valid.loc[:, 'shipping_0':'shipping_1'].values)
            X_val = hstack((X_item_condition_val, X_shipping_val, X_item_val,
                            X_brand_val, X_category_val, X_name_val)).tocsr()
            pred_val = 0.33 * self.lgbm_model_1.predict(X_val)
            pred_val += 0.34 * self.lgbm_model_2.predict(X_val)
            pred_val += 0.33 * self.ridge_model.predict(X_val)

            val_error = mean_squared_error(Y_val, pred_val, squared=False)
            print(f"Validation Error - {val_error:5.3f} \n")

        self.fit_performed = True

    def predict_df(self, dataset: pd.DataFrame):
        """
        Predicts the output for dataframe

        Parameters
        ----------
        data: pd.DataFrame
            The input data to be predicted on.

        Returns
        -------
        pred: np.ndarray
            The predicted price.
        """
        if self.fit_performed is False:
            raise AssertionError("Build the model with `fit` method and "
                                 "then use `predict_df` method.")
        X_name = self.feat_name.transform(dataset['name'])
        X_category = self.feat_category.transform(dataset['category_name'])
        X_brand = self.feat_brand.transform(dataset['brand_name'])
        X_item = self.feat_item_descp.transform(dataset['item_description'])
        X_item_condition = csr_matrix(
            dataset.loc[:, 'item_condition_id_1':'item_condition_id_5'].values)
        X_shipping = csr_matrix(dataset.loc[:,
                                            'shipping_0':'shipping_1'].values)
        X = hstack((X_item_condition, X_shipping, X_item, X_brand, X_category,
                    X_name)).tocsr()
        pred = 0.33 * self.lgbm_model_1.predict(X)
        pred += 0.34 * self.lgbm_model_2.predict(X)
        pred += 0.33 * self.ridge_model.predict(X)
        pred = np.expm1(pred)
        return pred

    def save(self, filename):
        """
        Saves the model weights in the required path in pickle format.

        Parameters
        ----------
        filename: str
            location to save the model.

        """
        if self.fit_performed is False:
            raise AssertionError("Build the model with `fit` method and "
                                 "then use `save` method.")
        with open(filename, 'wb') as f:
            pickle.dump(
                {
                    "name_featuriser": self.feat_name,
                    "category_featuriser": self.feat_category,
                    "brand_featuriser": self.feat_brand,
                    "item_desc_featuriser": self.feat_item_descp,
                    "ridge_model": self.ridge_model,
                    "lgbm_model_1": self.lgbm_model_1,
                    "lgbm_model_2": self.lgbm_model_2
                }, f)

    @classmethod
    def load(cls, filename):
        """
        Loads the model weights from the required path in pickle format.

        Parameters
        ----------
        filename: str
            location to return the model.

        Returns
        -------
        cls: PricePredictionModel
            The class object with loaded weights
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return cls(data["name_featuriser"],
                       data["category_featuriser"],
                       data["brand_featuriser"],
                       data["item_desc_featuriser"],
                       data["ridge_model"],
                       data["lgbm_model_1"],
                       data["lgbm_model_2"],
                       fit_performed=True)
