# flake8: noqa
from price_prediction.featuriser import NaiveFeaturiser
from price_prediction.model import PricePredictionModel
from price_prediction.predictor import Regressor
from price_prediction.utils import *
import os
from os.path import join
from flask import Flask
from flask import Flask, jsonify
from flask import request
import zipfile
import logging
import time

def create_app():
    """
    Instantiates the main flask instance, that creates the REST Server.
    It calls the Regressor pipeline and returns the prediction.

    Routes
    ------

    '/v1/price': Predicts the price of a product based of features.

    """
    app = Flask(__name__, instance_relative_config=True)
    featuriser_path = join(*[os.getcwd(), "weights", "featuriser.zip"])
    model_path = join(*[os.getcwd(), "weights", "model_weights.zip"])
    with zipfile.ZipFile(featuriser_path,"r") as zip_ref:
        zip_ref.extractall(join(*[os.getcwd(), "weights"]))
    with zipfile.ZipFile(model_path,"r") as zip_ref:
        zip_ref.extractall(join(*[os.getcwd(), "weights"]))

    pipeline = Regressor.load(
        feat_path=join(*[os.getcwd(), "weights", "featuriser.pkl"]),
        model_path=join(*[os.getcwd(), "weights", "model_weights.pkl"]))

    if os.path.isfile(join(*[os.getcwd(), "weights", "featuriser.pkl"])):
        os.remove(join(*[os.getcwd(), "weights", "featuriser.pkl"]))
    if os.path.isfile(join(*[os.getcwd(), "weights", "model_weights.pkl"])):
        os.remove(join(*[os.getcwd(), "weights", "model_weights.pkl"]))

    # Creating an object
    logger = logging.getLogger()


    @app.route('/v1/price', methods=['POST'])
    def predict():
        """
        Request
        -------
        >>> import requests
        >>> files = {
        ...   "name":"Hold Alyssa Frye Harness boots 12R, Sz 7",
        ...   "item_condition_id":3,
        ...   "category_name":"Women/Shoes/Boots",
        >>> }
        >>> requests.post("http://localhost:5000/v1/price", json=files).json()

        {'price': 44}
        """
        if request.method == 'POST':
            t1 = time.time()
            data = request.get_json()
            data = pipeline(data)
            print(f"Time taken for prediction {time.time()-t1}")
            return jsonify(data)

    return app
