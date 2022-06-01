import os
import tempfile
import pandas as pd
from price_prediction import NaiveFeaturiser


def test_save_load_featuriser():
    test_csv = pd.read_csv(
        os.path.join(*[os.getcwd(), "tests", "sample_train.csv"]))
    sample_req = test_csv.loc[0].to_dict()
    feat, test_csv, _ = NaiveFeaturiser.build_from_dataframe(test_csv)
    tmp_file = tempfile.NamedTemporaryFile()
    preds = feat(sample_req)
    feat.save(tmp_file.name)
    tmp_file.flush()
    new_feat = NaiveFeaturiser.load(tmp_file.name)
    new_preds = new_feat(sample_req)
    assert preds == new_preds
