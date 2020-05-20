import click
import logging

import yacomo
import yacomo.data
import yacomo.simulator

from yacomo.util import init_logging, log_error, log_warn, \
    log_debug, log_verbose, log_info, is_debug



@click.group()
@click.option('--debug', default=False, is_flag=True)
@click.option('--verbose', default=False, is_flag=True)
def main(debug, verbose):
    """Yacomo command line tool"""
    yacomo.util.init_logging(debug, verbose)

@main.group()
def model():
    pass
    
@model.command()
@click.option('--config-file',
              help='Training configuration file')
@click.option('--data-file',
              help='Data configuration file')
@click.option('--predictor-file',
              help='Output file for predictor parameters')
def train(config_file, data_file, predictor_file):
    yacomo.train(config_file, data_file, predictor_file)

@model.command()
@click.option('--predictor-file',
              help='Output file for predictor parameters')
@click.option('--num-days', type=int,
              help='Number of days to predict')
@click.option('--predictions-file',
              help='Output predictions file')
def predict(predictor_file, num_days, predictions_file):
    yacomo.predict(predictor_file, num_days, predictions_file)

    
@main.group()
def data():
    pass

@data.command()
@click.option('--config-file')
@click.option('--output-file')
def extract(config_file, output_file):
    yacomo.data.extract(config_file, output_file)

@data.command()
@click.option('--predictions-file')
@click.option('--data-file')
@click.option('--report-file')
def render(predictions_file, data_file, report_file):
    yacomo.data.render(predictions_file, data_file, report_file)


    ## For debugging
@main.command()
def debug():

    learned_params = {
        'r0_before': 2.82004334,
        'day_sd_start': 42.38768095,
        'r0_after': 0.7997493,
        'day_first_goner': 15.98899999,
        'sigmoid_param': 0.43402217
    }
    config = {'smoothing_window_size': 1,
              'days_to_death': 21,
              'days_contagious': 8}
    
    simulator = yacomo.simulator.SimAntelope(config)
    #simulator.set_parameters(**learned_params)
    pred = yacomo.simulator.Predictor(simulator, **learned_params)
    log_verbose(str(pred._simulator.parameters()))
    pred_estimate = pred.run(50)
    #log_verbose('pred_estimate: %s', str(pred_estimate))
    
    new_pred = yacomo.simulator.Predictor.from_parameters(pred.parameters())
    log_verbose(str(new_pred._simulator.parameters()))
    new_pred_estimate = new_pred.run(50)
    #log_verbose('new_pred_estimate: %s', str(new_pred_estimate))

    log_verbose('estimates:')
    for pair in zip(pred_estimate, new_pred_estimate):
        log_verbose(pair)
