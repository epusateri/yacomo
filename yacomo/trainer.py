import copy
import sys
import collections

import numpy as np
import scipy.optimize

import yacomo.simulator
import yacomo.objective
from yacomo.util import log_error, log_warn, log_info, log_verbose, log_debug, is_debug


class Trainer:
    pass


class TrApricot(Trainer):
    
    def __init__(self, config):
        self._simulator = yacomo.simulator.SimAntelope(config['simulator'])
        self._objective = yacomo.objective.ObjAmethyst(config['objective'],
                                                       self._simulator)
        self._niter = config['niter']
        self._minimizer_kwargs = config['minimizer_kwargs']
        self._op_bounds = config['bounds']

    _PARAMS = ['r0_before',
               'day_sd_start',
               'r0_after',
               'day_first_goner',
               'sigmoid_param']    
    def train(self, data):
        # TODO: Use an object to represent a group of predictors
        predictors = collections.defaultdict(dict)
        predictor_params = {
            'start_date': data['start_date'],
            'predictors': predictors
        }
        for region, region_data in data['daily_deaths'].items():
            for subregion, daily_deaths in region_data.items():
                log_info('Training predictor for %s', subregion)
                subregion_predictor = self._train_subregion(np.asarray(daily_deaths))
                predictors[region][subregion] = subregion_predictor.parameters()
        return predictor_params

    def _train_subregion(self, target_df):
        self._objective.set_target_df(target_df)

        bounds = [None]*len(self._PARAMS)
        for b in range(0, len(bounds)):
            bounds[b] = (self._op_bounds[self._PARAMS[b]]['min'],
                         self._op_bounds[self._PARAMS[b]]['max'])
            log_debug('bounds: %s', str(bounds))
            
        # Initialize parameters to mid-point between bounds
        x0 = []
        for b in bounds:
            x0.append((b[0] + b[1])/2.0)
        log_verbose('x0: %s', str(x0))

        def _accept_test(f_new, x_new, f_old, x_old):
            for i, x_i in enumerate(x_new):
                if x_i < bounds[i][0] or x_i > bounds[i][1]:
                    return False
            return True

        def _basinhopping_callback(x, f, accept):
            log_verbose("x: %s f: %f accept: %s", str(x), f, str(accept))

        minimizer_kwargs = copy.deepcopy(self._minimizer_kwargs)
        minimizer_kwargs['bounds'] = bounds
        log_verbose(x0)
        log_verbose(self._niter)
        log_verbose(minimizer_kwargs)
        log_verbose(self._objective._target_df)
        result = scipy.optimize.basinhopping(
            lambda x: self._objective.compute(*x),
            x0,
            niter = self._niter,
            minimizer_kwargs = minimizer_kwargs,
            callback = _basinhopping_callback,
            accept_test = _accept_test)

        learned_params = {}
        for p, param in enumerate(self._PARAMS):
            learned_params[param] = result.x[p]
        log_info('learned_params: %s', str(learned_params))

        return yacomo.simulator.Predictor(self._simulator, **learned_params)
