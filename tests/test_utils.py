# flake8: noqa
import pytest
import pandas as pd
import numpy as np
from price_prediction.utils import (handle_missing_inplace,
                                    build_crossvalidation_data, lower_case_df,
                                    cutting, check_dataframe_features,
                                    convert_dataframe_categorical,
                                    one_hot_encoding, clean_sentence)


def test_handle_missing_inplace():
    data = {
        'brand_name': ['Tom Selleck', np.NaN, 'Steve Madden', 'Louis Vuitton'],
        'item_description': ['very good', 'not bad', np.NaN, np.NaN]
    }
    df = pd.DataFrame(data)
    df = handle_missing_inplace(df)

    assert isinstance(df, pd.DataFrame)
    assert df['brand_name'].isna().sum() == 0
    assert df['item_description'].isna().sum() == 0
    assert np.all(df[df['brand_name'] == 'missing'].index == [1])
    assert np.all(df[df['item_description'] == 'missing'].index == [2, 3])


def test_build_crossvalidation_data():
    data = {
        'name': ['item1', 'item2', 'item3', 'item4', 'item5'],
        'brand_name': ['abcde', 'bcdea', 'cdeab', 'deabc', 'eabcd'],
        'item_description': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    split = 5
    train, val = build_crossvalidation_data(df, split)
    assert len(train) is split
    assert len(val) is split
    for tr, va in zip(train, val):
        assert isinstance(tr, pd.DataFrame)
        assert isinstance(va, pd.DataFrame)


def test_lower_case_df():
    data = {
        'name': ['ITEM1', 'iTeM2', 'item3', 'item4'],
        'brand_name': ['Tom Selleck', 'Pink', 'Steve Madden', 'Louis Vuitton'],
        'item_description': ['Very Good', 'not bad', 'OKKay', 'Decent']
    }
    df = pd.DataFrame(data)
    df = lower_case_df(df)
    assert isinstance(df, pd.DataFrame)
    assert np.all(df['name'].str.islower())
    assert np.all(df['brand_name'].str.islower())
    assert np.all(df['item_description'].str.islower())


def test_cutting():
    data = {'brand_name': ['Aa aa', 'bb cc', 'cc dd', 'dd ee']}
    df = pd.DataFrame(data)
    pop_brand = ['Aa aa', 'cc dd']
    df = cutting(df, pop_brand)
    assert isinstance(df, pd.DataFrame)
    assert np.all(df[df['brand_name'] == 'missing'].index == [1, 3])


def test_check_dataframe_features():
    data = {
        'name': ['a', 'b'],
        'brand_name': ['Aa aa', 'bb cc'],
        'item_condition': [1, 2]
    }

    required_features = ['name', 'brand_name', 'item_condition']
    df_type = [str, str, int]
    df = pd.DataFrame(data)
    check_dataframe_features(df, required_features, df_type)

    with pytest.raises(Exception) as e_info:
        required_features += ['shipping']
        check_dataframe_features(df, required_features, df_type)
        data = {
            'name': ['a', 'b'],
            'brand_name': ['Aa aa', 'bb cc'],
            'item_condition': ['1', '2']
        }
        required_features = ['name', 'brand_name', 'item_condition']
        df = pd.DataFrame(data)
        check_dataframe_features(df, required_features, df_type)


def test_convert_dataframe_categorical():
    data = {'item_condition': [1, 2, 3]}
    value_set = [1, 2, 3, 4, 5]
    field = 'item_condition'
    df = pd.DataFrame(data)
    df = convert_dataframe_categorical(df, field, value_set)
    assert isinstance(df, pd.DataFrame)
    assert np.all(df.columns == [f"item_condition_{i}" for i in range(1, 6)])


def test_clean_sentence():

    assert clean_sentence("verify the product at @twitter") \
        == "verify the product at twitter"
    assert clean_sentence("a neat product for sale at $40, buy immedieatly!!") \
        == "a neat product for sale at 40 buy immedieatly"
    assert clean_sentence("visit http://www.stevemadden.com for further enquires") \
        == "visit  for further enquires"


def test_one_hot_encoding():
    x = 1.
    allowable_set = [0., 1., 2.]
    assert one_hot_encoding(x, allowable_set) == [0, 1, 0]
    assert one_hot_encoding(x, allowable_set,
                            encode_unknown=True) == [0, 1, 0, 0]

    assert one_hot_encoding(x, allowable_set) == [0, 1, 0, 0]
    assert one_hot_encoding(x, allowable_set,
                            encode_unknown=True) == [0, 1, 0, 0]
    assert one_hot_encoding(-1, allowable_set,
                            encode_unknown=True) == [0, 0, 0, 1]
