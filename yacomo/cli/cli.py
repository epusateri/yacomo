import click
import logging

import yacomo
import yacomo.util
from yacomo.util import init_logging, log_error, log_warn, \
    log_debug, log_verbose, log_info, is_debug
import yacomo.data

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
    log_info("Test")
    log_verbose("ting")
