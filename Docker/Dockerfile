# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1
RUN pip3 install setuptools==60.1.0
RUN pip3 install -e .
RUN export FLASK_APP=price_prediction
RUN flake8 price_prediction --count
RUN pytest -v --cov=price_prediction
CMD [ "waitress-serve", "--call" , "price_prediction:create_app"]