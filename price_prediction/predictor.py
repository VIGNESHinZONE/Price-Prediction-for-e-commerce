from .featuriser import NaiveFeaturiser
from .model import PricePredictionModel

_required_fields = ['name', 'item_condition_id', 'category_name']
_required_fields_type = [str, int, str]

_optional_fields = ['brand_name', 'shipping', 'item_description']
_optional_fields_type = [str, int, str]


class Regressor(object):
    """
    This is the main class which encompasses the Featuriser and
    PredictionModel. Upon initilizing, this class could be used
    to directly called the API, with dictionary type request body.

        [1] We load the main featuriser in this class. `NaiveFeaturiser`
        [2] We load the main predictor in this class. `PricePredictionModel`
    """
    def __init__(self, data_featuriser: NaiveFeaturiser,
                 data_predictor: PricePredictionModel):
        """
        Parameters
        ----------
        data_featuriser: NaiveFeaturiser
            Class object for featuriser.
        data_predictor: PricePredictionModel
            Class object for prediction.
        """
        self.data_feat = data_featuriser
        self.data_pred = data_predictor

    def inspect_request(self, data: dict) -> dict:
        """
        Inspects the existence and types of certain fields.

        Parameters
        ----------
        data: dict
            The input data dictionary.

        Returns
        -------
        data: dict
            Resultant output dictionary.
        """
        for field, f_type in zip(_required_fields, _required_fields_type):
            if field not in data:
                return {"comment": f"{field} key not present in request"}
            if type(data[field]) is not f_type:
                return {
                    "comment":
                    f"{field} key should be of type {f_type}"
                    f", but instead got {type(data[field])}"
                }
            if field == 'item_condition_id':
                if (data[field] < 1 or data[field] > 5):
                    return {
                        "comment": ("item_condition_id key should be within "
                                    "range [1, 5]")
                    }

        for field, f_type in zip(_optional_fields, _optional_fields_type):
            if field in data:
                if type(data[field]) is not f_type:
                    return {
                        "comment":
                        f"{field} key should be of type {f_type}"
                        f", but instead got {type(data[field])}"
                    }
                if field == 'item_condition_id':
                    if (data[field] < 1 or data[field] > 5):
                        return {
                            "comment":
                            ("item_condition_id key should be within "
                             "range [1, 5]")
                        }
                if field == 'shipping':
                    if (data[field] < 0 or data[field] > 1):
                        return {
                            "comment": ("shipping key should be "
                                        "within range [0, 1]")
                        }

            else:
                data[field] = "" if f_type is str else -1

        return data

    def __call__(self, data: dict) -> dict:
        """
        This enables class the featuriser and predictor. And it
        then returns the appropriate resultant.

        Parameters
        ----------
        data: dict
            The input data dictionary.

        Returns
        -------
        data: dict
            Resultant output dictionary.
        """

        data = self.inspect_request(data)
        if "comment" in data:
            return data
        data = self.data_feat(data)
        price = self.data_pred.predict(data)
        return {"price": max(3, int(price))}

    @classmethod
    def load(cls, feat_path, model_path):
        """
        Loads the Featuriser & Predictor from the required path in
        pickle format.

        Parameters
        ----------
        feat_path: str
            location to return the featuriser.
        model_path: str
            location to return the model.

        Returns
        -------
        cls: Regressor
            The class object with loaded weights
        """
        data_featuriser = NaiveFeaturiser.load(feat_path)
        data_pred = PricePredictionModel.load(model_path)
        return cls(data_featuriser, data_pred)
