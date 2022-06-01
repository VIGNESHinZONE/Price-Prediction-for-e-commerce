import os
import json
import pandas as pd
import numpy as np
from price_prediction import create_app


def test_simple_request():
    flask_app = create_app()

    with flask_app.test_client() as test_client:
        files = {
            "name": "Hold Alyssa Frye Harness boots 12R, Sz 7",
            "item_condition_id": 3,
            "category_name": "Women/Shoes/Boots",
        }

        response = test_client.post('/v1/price', json=files)
        res = response.data.decode('utf8')
        res = json.loads(res)
        assert response.status_code == 200
        assert isinstance(res, dict)
        assert 'price' in res
        assert 'comment' not in res
        assert res['price'] == 44


def test_multiple_request_types():
    flask_app = create_app()

    with flask_app.test_client() as test_client:

        test_csv = pd.read_csv(
            os.path.join(*[os.getcwd(), "tests", "sample_train.csv"]))
        for i in range(50, 60):
            files = test_csv.loc[i].to_dict()
            files = {
                key: files[key]
                for key in files if files[key] is not np.NaN
            }
            response = test_client.post('/v1/price', json=files)
            res = response.data.decode('utf8')
            res = json.loads(res)
            assert response.status_code == 200
            assert isinstance(res, dict)
            assert 'price' in res
            assert 'comment' not in res
            assert res['price'] >= 3


def test_false_request_types():
    flask_app = create_app()

    with flask_app.test_client() as test_client:
        files = {
            'name': 'New Michael Kors Aileen Boot Sz. 9.5',
            'item_condition_id': '1',
            'category_name': 'Women/Shoes/Boots',
            'brand_name': 'Michael Kors',
            'price': '75',
            'shipping': '0',
        }
        response = test_client.post('/v1/price', json=files)
        res = response.data.decode('utf8')
        res = json.loads(res)
        # statement = ("item_condition_id key should be of type"
        # " <class 'int'>, but instead got <class 'str'>")
        assert response.status_code == 200
        assert isinstance(res, dict)
        assert 'price' not in res
        assert 'comment' in res
        assert res['comment'] == ("item_condition_id key should be of type"
                                  " <class 'int'>, but instead got"
                                  " <class 'str'>")

        files = {
            'name': 'New Michael Kors Aileen Boot Sz. 9.5',
            'item_condition_id': 1,
            'category_name': 'Women/Shoes/Boots',
            'brand_name': 'Michael Kors',
            'price': 75,
            'shipping': 5,
        }
        response = test_client.post('/v1/price', json=files)
        res = response.data.decode('utf8')
        res = json.loads(res)
        # statement = ("item_condition_id key should be of type"
        # " <class 'int'>, but instead got <class 'str'>")
        assert response.status_code == 200
        assert isinstance(res, dict)
        assert 'price' not in res
        assert 'comment' in res
        assert res['comment'] == ("shipping key should be "
                                  "within range [0, 1]")

        files = {
            'name': 'New Michael Kors Aileen Boot Sz. 9.5',
            'item_condition_id': 7,
            'category_name': 'Women/Shoes/Boots',
            'brand_name': 'Michael Kors',
            'price': 75,
            'shipping': 1,
        }
        response = test_client.post('/v1/price', json=files)
        res = response.data.decode('utf8')
        res = json.loads(res)
        # statement = ("item_condition_id key should be of type"
        # " <class 'int'>, but instead got <class 'str'>")
        assert response.status_code == 200
        assert isinstance(res, dict)
        assert 'price' not in res
        assert 'comment' in res
        assert res['comment'] == ("item_condition_id key should be within "
                                  "range [1, 5]")
