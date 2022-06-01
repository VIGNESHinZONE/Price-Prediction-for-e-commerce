# Price Prediction Assignment
by Name - Vignesh Venkataraman

The task is to build a REST API server that can recommend the prices at which products
can be sold via the basis of input features like `['name', 'brand_name', 'description' ...]` .

---

# File Structure of the project

1. data - Consist of the input csv files used for training and prediction.
2. Docker - Consist of the Docker file to build the image
3. notebooks - Experiments & different methodologies used while building the models.
4. price_prediction - The main API consist of all the code to run the server and execute the models
5. tests - Consist of all the tests to run the API server. It includes tests for both model and flask server. These get executed using the `Pytest` library.
6. weights - Consists of all the model weights used for prediction. They are saved in zip format. There is no need to unzip them.


7. setup.py - For properly building the project
8. submission.csv - The output prediction of `data/mercari_test.csv` files.

---

# Building the project

## 1. With Docker

Here are all the tasks we can perform - 

1. Build the docker image, which automatically runs the unittests
    ```
    docker build --tag production . -f Docker/Dockerfile
    ```

2. Running the server directly, using the container built above.
    ```
    docker run --name production_server -d -p 8080:8080 production
    ```
    [see here](https://github.com/m-rec/547fc32ae0e26fbd6ebc5b4193c93468965a80dd#ways-to-check-the-api-server-running-properly-) to run a small test to see if the server is working fine.
        

3. To execute commands manually, run the container in interactive mode

    ```
    docker run -it --entrypoint=/bin/bash -p 8080:8080 production
    ```

    * To run pytest & flake tests
        ```
        flake8 price_prediction --count
        pytest -v --cov=price_prediction
        ```

    * To start the server - 
        ```
        waitress-serve --call 'price_prediction:create_app'
        ```
        [see here](https://github.com/m-rec/547fc32ae0e26fbd6ebc5b4193c93468965a80dd#ways-to-check-the-api-server-running-properly-) to run a small test to see if the server is working fine.

    * To run the training code - 
        ```
        python price_prediction/train.py
        ```


## 2. Without Docker, through conda or venv environments- 
1. Have any python virtual environment installed, and run the following commands to build the project
    ```
    pip install setuptools==60.1.0
    pip install -e .
    export FLASK_APP=price_prediction
    ``` 
2. To execute pytest and flake8 tests - 
    ```
    flake8 price_prediction --count
    pytest -v --cov=price_prediction
    ```
3. To start the server - 
    ```
    waitress-serve --call 'price_prediction:create_app'
    ```
    [see here](https://github.com/m-rec/547fc32ae0e26fbd6ebc5b4193c93468965a80dd#ways-to-check-the-api-server-running-properly-) to run a small test to see if the server is working fine.

4. To run the training code - 
    ```
    python price_prediction/train.py
    ```

---

# Training Approaches
    Details can be found in [train_methodlogy.md]()

---

# Ways To check the API Server running properly-
1. Run this command in the terminal

```bash
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"name": "Hold Alyssa Frye Harness boots 12R, Sz 7", "item_condition_id": 3, "category_name": "Women/Shoes/Boots"}' \
  http://0.0.0.0:8080/v1/price
```

2. Open another python terminal and execute this script

```python
import requests
files = {
  "name": "Hold Alyssa Frye Harness boots 12R, Sz 7",
  "item_condition_id": 3,
  "category_name": "Women/Shoes/Boots",
}
print(requests.post("http://0.0.0.0:8080/v1/price", json=files).json())
```
