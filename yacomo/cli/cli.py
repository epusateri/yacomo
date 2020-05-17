import click
import logging

import yacomo
import yacomo.util

@click.group()
@click.option('--debug', default=False, is_flag=True)
@click.option('--verbose', default=False, is_flag=True)
def main(debug, verbose):
    """Yacomo command line tool"""
    util.init_logging(debug, verbose)

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
