from loguru import logger
import pytest
from ultralytics import YOLO
import numpy as np 
import pandas as pd

def test_val():

    model = YOLO('/tmp/trains/710/output/best.pt')  # load an official model        
    # Validate the model
    metrics = model.val(data="/tmp/trains/710/dataset/data.yaml", split="test", save_json=True, conf=.4)
    # metrics = model.val(data="/tmp/trains/710/dataset/test")
    logger.info(metrics)


def test_val_test():
    model = YOLO('/tmp/trains/710/output/best.pt')  # load an official model        
    # Validate the model
    metrics = model.test(data="/tmp/trains/710/dataset/data.yaml", split="test", save_json=True, conf=.4)
    # metrics = model.val(data="/tmp/trains/710/dataset/test")
    logger.info(metrics)
    test_dict = {}
    for k, v in metrics.results_dict.items():
        test_dict[k] = np.array([v])
    test_rs_df = pd.DataFrame(test_dict)
    test_rs_df.to_csv('/tmp/test_result.csv', index=False)


if __name__ == '__main__':
    test_val_test()
    # test_val()