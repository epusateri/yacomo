#!/bin/bash

# git rev-parse HEAD > hash

# curl -O https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv

# yacomo \
#     --verbose \
#     data \
#     extract \
#     --config-file data.yaml \
#     --output-file data.json

# yacomo \
#     --verbose \
#     model \
#     train \
#     --data-file data.json \
#     --config-file train.yaml \
#     --predictor-file predictors.json

# yacomo \
#     --verbose \
#     model \
#     predict \
#     --predictor-file predictors.json \
#     --num-days 200 \
#     --predictions-file predictions.json

yacomo \
    --verbose \
    data \
    render \
    --predictions-file predictions.json \
    --data-file data.json \
    --report-file report.pdf

