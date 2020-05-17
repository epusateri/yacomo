import yaml
import numpy as np

import yacomo
import yacomo.simulator
import yacomo.objective
import yacomo.optimizer
import yacomo.trainer


def train(config_fn, data_fn, predictor_fn):

    with open(config_fn, encoding='utf-8') as config_fh:
        config = yaml.load(config_fh)

    target_df = np.arange(1, 150)
            
    # Create trainer
    trainer = yacomo.trainer.TrApricot(config['trainer'])

    # Train predictor
    predictor = trainer.train(target_df)
    
    with open(predictor_fn, 'w', encoding='utf-8') as predictor_fh:
        predictor.save(predictor_fh)
