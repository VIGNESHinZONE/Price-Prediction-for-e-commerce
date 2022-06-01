import pandas as pd
import re


def handle_missing_inplace(dataset: pd.DataFrame):
    """
    Changes missing values in pandas dataframe with fields
    ['brand_name', 'item_description'].

    Parameters
    ----------
    dataset: pd.DataFrame
        Dataframe to processs.

    Returns
    ----------
    dataset: pd.DataFrame
        Resultant output.
    """
    dataset['brand_missing'] = dataset['brand_name'].isna().astype(int)
    dataset['descp_missing'] = dataset['item_description'].isna().astype(int)

    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)
    return dataset


def build_crossvalidation_data(dataset: pd.DataFrame, split: int = 5):
    """
    Used for building k-fold cross validation datasets.

    Parameters
    ----------
    dataset: pd.DataFrame
        Dataframe to processs.
    split: int
        number of splits.

    Returns
    ----------
    train_datasets: List[pd.DataFrame]
        Train datasets in split folds.
    test_datasets: List[pd.DataFrame]
        Corresponding Test Datasets.
    """

    from sklearn.model_selection import KFold
    train_datasets = []
    test_datasets = []
    skf = KFold(n_splits=split, shuffle=True, random_state=402)
    skf.get_n_splits(dataset)
    for train_index, test_index in skf.split(dataset):
        train = dataset.iloc[train_index]
        validation = dataset.iloc[test_index]
        train_datasets.append(train)
        test_datasets.append(validation)
    return train_datasets, test_datasets


def lower_case_df(dataset: pd.DataFrame):
    """
    Lower case ['name', 'brand_name', 'item_description'] features.

    Parameters
    ----------
    dataset: pd.DataFrame
        Dataframe to processs.

    Returns
    ----------
    dataset: pd.DataFrame
        Resultant processed dataframe.
    """
    dataset['name'] = dataset['name'].str.lower()
    dataset['brand_name'] = dataset['brand_name'].str.lower()
    dataset['item_description'] = dataset['item_description'].str.lower()
    return dataset


def cutting(dataset: pd.DataFrame, pop_brand: list):
    """
    Changing ['brand_name'] not in pop_brand to 'missing' values.

    Parameters
    ----------
    dataset: pd.DataFrame
        Dataframe to processs.
    pop_brand: List[str]
        List of pop_brand.

    Returns
    ----------
    dataset: pd.DataFrame
        Resultant processed dataframe.
    """
    dataset['brand_name'] = dataset['brand_name'] \
        .apply(lambda x: x if x in pop_brand else "missing")
    return dataset


def check_dataframe_features(df: pd.DataFrame, required_features: list,
                             df_type: list):
    """
    Checks if required fields are present in Dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to processs.
    required_features: List[str]
        List of fields.
    df_type: List[types]
        List of data types.
    """
    for feature in required_features:
        try:
            assert feature in df
        except KeyError:
            raise KeyError(f"{feature} not present in {df_type} dataframe")


def convert_dataframe_categorical(dataset: pd.DataFrame, field: str,
                                  value_set: list):
    """
    Checks if required fields are present in Dataframe.

    Parameters
    ----------
    dataset: pd.DataFrame
        Dataframe to processs.
    field: str
        fields string.
    value_set: list
        List of values.

    Returns
    ----------
    dataset: pd.DataFrame
        Resultant processed dataframe.
    """
    from pandas.api.types import CategoricalDtype
    cat_type = CategoricalDtype(categories=value_set, ordered=True)
    dataset[field] = dataset[field].astype(cat_type)
    dataset = pd.get_dummies(dataset, columns=[field])
    return dataset


def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """
    One-hot encoding.

    Parameters
    ----------
    x:
        Value to encode.
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element.

    Returns
    -------
    list:
        List of boolean values where at most one value is True.
    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))


def clean_sentence(text):
    """
    Cleans a input string.

    Parameters
    ----------
    text:
        input string to process.

    Returns
    -------
    test:
        Resultant output string.
    """
    text = re.sub(r'(\w+:\/\/\S+)|http.+?', "", text)
    text = re.sub(r'(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])', '', text.lower())
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)

    return text.strip()
