import json
import yaml
import collections
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt

import numpy as np
import pandas as pd

from yacomo.util import log_error, log_warn, log_debug, log_verbose, log_info, is_debug

_JHU_US_FIRST_DAY_FIELD = 12
def _extract_from_jhu(data_fn, config):
    region_col = config['region_column']
    subregion_col = config['subregion_column']
    subregion_filter = config['subregion_filter']
    log_debug(subregion_filter)

    # TODO: This can likely be done more elegantly and mostly in pandas
    df = pd.read_csv(data_fn, quotechar='"', skipinitialspace=True)

    # HACK.  Sometimes the cumulative death numbers are not increasing
    df = df[df['Admin2'] != 'Unassigned']

    start_date =df.columns[_JHU_US_FIRST_DAY_FIELD + 1]
    subregion_df = df.groupby([region_col, subregion_col]).sum().reset_index().to_numpy()

    cum_deaths_df = subregion_df[:,12:]
    offset_cum_deaths_df = cum_deaths_df[:, 1:]
    daily_deaths_df = offset_cum_deaths_df - cum_deaths_df[:, :-1]

    # Construct data dictionary
    output_dict = {'start_date': start_date,
                   'daily_deaths': collections.defaultdict(collections.defaultdict)}
    for r in (range(subregion_df.shape[0])):
        region = subregion_df[r][0]
        subregion = subregion_df[r][1]
        log_debug(subregion)
        if subregion_filter and subregion not in subregion_filter:
            continue

        dd = daily_deaths_df[r].tolist()
        output_dict['daily_deaths'][region][subregion] = dd

    return output_dict

def extract(config_fn, output_fn):
    with open(config_fn, encoding='utf-8') as config_fh:
        config = yaml.load(config_fh)

    if config['type'] == 'jhu_deaths_US':
        output_dict = _extract_from_jhu(config['data_file'], config['extract'])

    with open(output_fn, 'w', encoding='utf-8') as output_file:
        json.dump(output_dict, output_file, indent=4)

def _format_subplot(ax, y_max):
    ytick_interval = int(y_max/10/5)*5
    yticks = np.arange(0, y_max, ytick_interval)
    ax.set_yticks(yticks)
    for label in ax.get_yticklabels():
        label.set_fontsize(5)

# TODO: Refactor/cleanup
def _build_subplots(ax, region,
                    daily_data, cum_data,
                    daily_pred, cum_pred,
                    date_labels):

    ax[0].set_title(region)
    ax[0].plot(daily_data)
    ax[0].plot(daily_pred, linestyle=':')
    ax[0].set_xlabel('day')
    ax[0].set_ylabel('daily deaths')

    y_max = max(daily_pred + daily_data)
    now = len(daily_data)
    two_weeks_out = len(daily_data) + 14

    ax[0].axvline(now, color='grey', linestyle='--')
    ax[0].axhline(daily_pred[now], color='grey', linestyle='--')

    y_max = max(daily_pred + daily_data)
    two_weeks_out = len(daily_data) + 14
    ax[0].axvline(two_weeks_out, color='grey', linestyle='--')
    ax[0].axhline(daily_pred[two_weeks_out], color='grey', linestyle='--')

    _format_subplot(ax[0], y_max)
    twin = ax[0].twinx()
    twin.set_ylim(ax[0].get_ylim())
    twin.set_yticks([0, daily_pred[now], daily_pred[two_weeks_out]])
    for label in twin.get_yticklabels():
        label.set_fontsize(8)
        
    xtick_indices = np.arange(0, len(daily_pred), 14)
    xtick_labels = date_labels[0::14]
    log_verbose(xtick_labels)
    ax[0].set_xticks(xtick_indices)
    ax[0].set_xticklabels(xtick_labels)

    for label in ax[0].get_xticklabels():
        label.set_fontsize(5)


    ###
    ax[1].set_title(region)
    ax[1].plot(cum_data)
    ax[1].plot(cum_pred, linestyle=':')
    ax[1].set_xlabel('day')
    ax[1].set_ylabel('cumulative deaths')
    ax[1].set_yticks(np.arange(0, max(cum_data), 500))

    y_max = max(cum_pred + cum_data)
    now = len(cum_data)
    two_weeks_out = len(daily_data) + 14

    ax[1].axvline(now, color='grey', linestyle='--')
    ax[1].axhline(cum_pred[now], color='grey', linestyle='--')

    ax[1].axvline(two_weeks_out, color='grey', linestyle='--')
    ax[1].axhline(cum_pred[two_weeks_out], color='grey', linestyle='--')
    ax[1].axhline(cum_pred[-1], color='grey', linestyle='--')

    _format_subplot(ax[1], y_max)
    twin = ax[1].twinx()
    twin.set_ylim(ax[1].get_ylim())
    twin.set_yticks([0, cum_pred[now], cum_pred[two_weeks_out], cum_pred[-1]])
    for label in twin.get_yticklabels():
        label.set_fontsize(8)

    xtick_indices = np.arange(0, len(daily_pred), 14)
    xtick_labels = date_labels[0::14]
    log_verbose(xtick_labels)
    ax[1].set_xticks(xtick_indices)
    ax[1].set_xticklabels(xtick_labels)

    for label in ax[1].get_xticklabels():
        label.set_fontsize(5)


def render(predictions_fn, data_fn, report_fn):
    with open(predictions_fn, encoding='utf-8') as predictions_file:
        predictions = json.load(predictions_file)
    with open(data_fn, encoding='utf-8') as data_file:
        data = json.load(data_file)

    # Assert that start dates are the same

    # Get total region data
    num_subregions = 0
    for region, region_data in data['daily_deaths'].items():
        any_region = region
        any_subregion = list(region_data.keys())[0]
        region_daily_deaths = np.zeros(len(data['daily_deaths'][region][any_subregion]))
        num_subregions += len(region_data)
        for subregion, subregion_data in region_data.items():
            region_daily_deaths += np.asarray(subregion_data)
        data['daily_deaths'][region]['REGION'] = region_daily_deaths.tolist()
        num_subregions += 1
        
    num_days = len(predictions['data'][any_region][any_subregion]['daily_deaths'])
    start_date_str = predictions['start_date']
    curr_date = dt.datetime.strptime(start_date_str, '%m/%d/%y')
    date_labels = [curr_date.strftime('%Y-%m-%d')]
    for d in range(1, num_days):
        curr_date += dt.timedelta(days=1)
        date_labels.append((curr_date).strftime('%m-%d'))
        
    fig, axs = plt.subplots(num_subregions, 2, figsize=(3*num_subregions, 11))
    for region, region_data in predictions['data'].items():

        s = 0
        subregion = 'REGION'
        subregion_data = region_data[subregion]
        data_daily_deaths = data['daily_deaths'][region][subregion]
        data_cum_deaths = np.asarray(data_daily_deaths).cumsum().tolist()

        _build_subplots(axs[s], region,
                        data_daily_deaths, data_cum_deaths,
                        subregion_data['daily_deaths'], subregion_data['cumulative_deaths'],
                        date_labels)
        s+= 1

        for subregion, subregion_data in region_data.items():
            if subregion == 'REGION':
                continue

            data_daily_deaths = data['daily_deaths'][region][subregion]
            data_cum_deaths = np.asarray(data_daily_deaths).cumsum().tolist()

            _build_subplots(axs[s], subregion,
                            data_daily_deaths, data_cum_deaths,
                            subregion_data['daily_deaths'], subregion_data['cumulative_deaths'],
                            date_labels)
            s += 1

    plt.tight_layout()
    plt.savefig(report_fn, format='pdf')
