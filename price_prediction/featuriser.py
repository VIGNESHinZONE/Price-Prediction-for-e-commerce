import numpy as np
import pickle
from .utils import (check_dataframe_features, convert_dataframe_categorical,
                    handle_missing_inplace, one_hot_encoding, cutting,
                    lower_case_df, clean_sentence)

DEFAULT_CONDITION_IDS = [1, 2, 3, 4, 5]
DEFAULT_SHIPPING_IDS = [0, 1]


class NaiveFeaturiser(object):
    """
    The Featuriser class which preprocess inputs, which can
    then be appropriately fed into the Model class.

    This class consists of pre-processes which are common for
    both train and test data.

    Here is the list of preprocesses done in this class
    built from dataframe-
        [1] Checks if dataframe is in appropriate format with all
            required fields.
        [2] Removes all missing values in the dataframe to 'missing'
            value.
        [3] Lower case the text fields ['name', 'brand_name',
            'item_description'] and normalizes the text.
        [4] We build a list of popular brands that have occured a minimum
            of 5 times and convert the rest of brand names to 'missing'.
        [5] Converts ['item_condition_id', 'shipping'] fields to one hot
            encodings.
    """
    def __init__(self,
                 brand_names,
                 item_condition_set=DEFAULT_CONDITION_IDS,
                 shipping_set=DEFAULT_SHIPPING_IDS):
        """
        Parameters
        ----------
        brand_names: List[str]
            List of all popular brand names.
        item_condition_set: List[int]
            List of all possible item_condition_id values
            in request. Default to [1, 2, 3, 4, 5].
        shipping_set: List[int]
            List of all possible shipping values
            in request. Default to [0, 1].
        """
        self.popular_brand_names = brand_names
        self.item_condition_set = item_condition_set
        self.shipping_set = shipping_set

    def __call__(self, data: dict) -> dict:
        """
        Calls the featuriser methods, which processes a single
        dictionary object.

        Parameters
        ----------
        data: dict
            A dictionary object with induvidual features.

        Returns
        -------
        data: dict
            A dictionary object with processed features.
        """
        data = self._clean_request(data)
        if data['brand_name'] not in self.popular_brand_names:
            data['brand_name'] = 'missing'

        data['item_condition_id'] = \
            np.array(one_hot_encoding(
                data['item_condition_id'],
                self.item_condition_set
            ), dtype=int)

        data['shipping'] = \
            np.array(one_hot_encoding(
                data['shipping'],
                self.shipping_set
            ), dtype=int)

        return data

    @classmethod
    def build_from_dataframe(cls, df_train, df_test=None):
        """
        Builds the featuriser using the dataframe.

        Parameters
        ----------
        df_train: pd.Dataframe
            The input dataframe used for building the featuriser.
        df_test: pd.Dataframe
            The input dataframe used for processing.
            Default to None

        Returns
        -------
        cls: Naive_Featuriser
            Class object for featuriser
        df_train: pd.Dataframe
            Processed output of df_train
        df_test: pd.Dataframe
            Processed output of df_test
        """
        required_features = [
            "name", "item_condition_id", "category_name", "brand_name",
            "shipping", "seller_id", "item_description"
        ]
        check_dataframe_features(df_train, required_features, "df_train")
        df_train = handle_missing_inplace(df_train)
        df_train = lower_case_df(df_train)
        df_train["name"] = df_train["name"].apply(clean_sentence)
        df_train["item_description"] = \
            df_train["item_description"].apply(clean_sentence)

        popular_brand = df_train['brand_name'] \
            .value_counts() \
            .loc[lambda x: x.index != 'missing'] \
            .loc[lambda x: x >= 5].to_dict()
        df_train = cutting(df_train, popular_brand)
        df_train = convert_dataframe_categorical(df_train, 'item_condition_id',
                                                 DEFAULT_CONDITION_IDS)
        df_train = convert_dataframe_categorical(df_train, 'shipping',
                                                 DEFAULT_SHIPPING_IDS)

        if df_test is not None:
            check_dataframe_features(df_test, required_features, "df_test")
            df_test = handle_missing_inplace(df_test)
            df_test = lower_case_df(df_test)
            df_test["name"] = df_test["name"].apply(clean_sentence)
            df_test["item_description"] = \
                df_test["item_description"].apply(clean_sentence)

            df_test = cutting(df_test, popular_brand)
            df_test = convert_dataframe_categorical(df_test,
                                                    'item_condition_id',
                                                    DEFAULT_CONDITION_IDS)
            df_test = convert_dataframe_categorical(df_test, 'shipping',
                                                    DEFAULT_SHIPPING_IDS)

        return cls(popular_brand), df_train, df_test

    def save(self, filename):
        """
        Saves the Featuriser in the required path in pickle format.

        Parameters
        ----------
        filename: str
            location to save the model.

        """
        with open(filename, 'wb') as f:
            pickle.dump(
                {
                    "popular_brands": self.popular_brand_names,
                    "item_conditions": self.item_condition_set,
                    "shipping_ids": self.shipping_set
                }, f)

    @classmethod
    def load(cls, filename):
        """
        Loads the Featuriser from the required path in pickle format.

        Parameters
        ----------
        filename: str
            location to return the model.

        Returns
        -------
        cls: Naive_Featuriser
            The class object with loaded weights
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return cls(data["popular_brands"], data["item_conditions"],
                       data["shipping_ids"])

    def _clean_request(self, data):
        """
        It performs a basic cleaning for all text fields

        Parameters
        ----------
        data: dict
            request body

        Returns
        -------
        data: dict
            processed request body
        """
        if data['brand_name'] == "":
            data['brand_name'] = 'missing'

        if data['item_description'] == "":
            data['brand_name'] = 'missing'

        data['name'] = clean_sentence(data['name'].lower())
        data['brand_name'] = data['brand_name'].lower()
        data['item_description'] = clean_sentence(
            data['item_description'].lower())
        return data
