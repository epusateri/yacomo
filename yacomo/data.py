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
    to_drop = [c for c in df.columns[:_JHU_US_FIRST_DAY_FIELD] if c != region_col and c!= subregion_col]
    df = df.drop(columns=to_drop)

    if 'end_date' in config:
        end_date = dt.datetime.strptime(config['end_date'], '%Y-%m-%d')
        to_drop = [c for c in df.columns[2:] if dt.datetime.strptime(c, '%m/%d/%y') > end_date]
        log_verbose(to_drop)
        df = df.drop(columns=to_drop)

    # TODO: Validate data

    start_date = df.columns[3]
    log_verbose('start_date: %s', start_date)

    subregion_df = df.groupby([region_col, subregion_col]).sum().reset_index().to_numpy()
    cum_deaths_df = subregion_df[:,2:]
        
    offset_cum_deaths_df = cum_deaths_df[:, 1:]
    daily_deaths_df = offset_cum_deaths_df - cum_deaths_df[:, :-1]

    # Construct data dictionary
    output_dict = {'start_date': start_date,
                   'daily_deaths': collections.defaultdict(collections.defaultdict)}
    for r in (range(subregion_df.shape[0])):
        region = subregion_df[r][0]
        subregion = subregion_df[r][1]
        log_verbose('Reading data for %s', subregion)
        if subregion_filter and subregion not in subregion_filter:
            continue
        total = daily_deaths_df[r].sum()
        if total < config['subregion_min_deaths']:
            log_info('Skipping %s', subregion)
            
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
    ytick_interval = max(int(y_max/10/5)*5, 1)
    yticks = np.arange(0, y_max, ytick_interval)
    ax.set_yticks(yticks)
    for label in ax.get_yticklabels():
        label.set_fontsize(5)

# TODO: Refactor/cleanup
def _build_subplots(ax, region,
                    daily_data, cum_data,
                    daily_pred, cum_pred,
                    date_labels):

    log_verbose('Creating subplots for %s', region)

    # Daily
    ax[0].set_title(region)
    ax[0].plot(daily_data, linestyle='None', marker='o', markersize=2)
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
    ax[0].set_xticks(xtick_indices)
    ax[0].set_xticklabels(xtick_labels)

    for label in ax[0].get_xticklabels():
        label.set_fontsize(4)

    # Cumulative
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
    ax[1].set_xticks(xtick_indices)
    ax[1].set_xticklabels(xtick_labels)

    for label in ax[1].get_xticklabels():
        label.set_fontsize(4)


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
        for subregion, subregion_data in region_data.items():
            region_daily_deaths += np.asarray(subregion_data)
            num_subregions += 1
        data['daily_deaths'][region]['REGION'] = region_daily_deaths.tolist()
        num_subregions += 1
        
    num_days = len(predictions['data'][any_region][any_subregion]['daily_deaths'])
    start_date_str = predictions['start_date']
    curr_date = dt.datetime.strptime(start_date_str, '%m/%d/%y')
    log_verbose('curr_date: %s', curr_date)
    date_labels = [curr_date.strftime('%m-%d')]
    for d in range(1, num_days+1):
        curr_date += dt.timedelta(days=1)
        date_labels.append((curr_date).strftime('%m-%d'))

    log_verbose(date_labels)
    log_verbose(len(region_daily_deaths))
    log_verbose(date_labels[len(region_daily_deaths)])
        
    fig, axs = plt.subplots(num_subregions, 2, figsize=(8.5, 3.5*num_subregions))
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
