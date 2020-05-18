import yaml
import json
import numpy as np
import collections

import yacomo
import yacomo.trainer
import yacomo.simulator


def train(config_fn, data_fn, predictor_fn):
    with open(config_fn, encoding='utf-8') as config_fh:
        config = yaml.load(config_fh)

    with open(data_fn, encoding='utf-8') as data_file:
        data = json.load(data_file)
            
    # Create trainer
    trainer = yacomo.trainer.TrApricot(config['trainer'])

    # Train predictors
    predictor_params = trainer.train(data)

    # Save predictor parameters
    with open(predictor_fn, 'w', encoding='utf-8') as predictor_file:
        json.dump(predictor_params, predictor_file, indent=4)

def predict(predictor_fn, n_days, predictions_fn):
    with open(predictor_fn, encoding='utf-8') as predictor_file:       
        predictors = json.load(predictor_file)

    # TODO: Move this into it's own module/class
    data = {}
    output_dict = {
        'start_date': predictors['start_date'],
        'data': data
    }
    predictor_params = predictors['predictors']
    for region, region_data in predictor_params.items():

        region_preds = collections.defaultdict(collections.defaultdict)
        data[region] = region_preds

        region_daily_deaths = np.zeros(n_days)        
        for subregion, daily_deaths in region_data.items():
            predictor = yacomo.simulator.Predictor.from_parameters(
                predictor_params[region][subregion])
            daily_deaths = predictor.run(n_days)
            region_preds[subregion]['daily_deaths'] = daily_deaths
            region_preds[subregion]['cumulative_deaths'] = np.cumsum(daily_deaths).tolist()
            region_daily_deaths += np.asarray(daily_deaths)
            
        region_preds['REGION']['daily_deaths'] = region_daily_deaths.tolist()
        region_preds['REGION']['cumulative_deaths'] = np.cumsum(region_daily_deaths).tolist()
            
    with open(predictions_fn, 'w', encoding='utf-8') as predictions_file:
        json.dump(output_dict, predictions_file, indent=4)
