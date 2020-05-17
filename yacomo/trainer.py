import copy
import sys

import scipy.optimize

import yacomo.predictor
import yacomo.simulator
from yacomo.util import log_error, log_warn, log_info, log_verbose, is_debug


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
    
    # def _basinhopping_callback(x, f, accept):
    #     #log_info("x: %s f: %f accept: %b" % (str(x), f, accept))
    #     print("x: %s f: %f accept: %b" % (str(x), f, accept))
    #     sys.stdout.flush()

    # def _accept_test(self, f_new, x_new, f_old, x_old, bounds):
    #     log_debug('x_new: %s', x_new)
    #     for i, x_i in enumerate(x_new):
    #         if x_i < bounds[i][0] or x_i > bounds[i][1]:
    #             return False
    #     log_debug('accepted')
    #     return True
    
    def train(self, target_df):
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
        log_debug('x0: %s', str(x0))

        obj = self._objective.compute(*x0)
        log_debug('obj: %f', obj)

        def _accept_test(f_new, x_new, f_old, x_old):
            print('x_new: %s', x_new)
            sys.stdout.flush()
            for i, x_i in enumerate(x_new):
                if x_i < bounds[i][0] or x_i > bounds[i][1]:
                    return False
                log_debug('accepted')
            return True

        def _basinhopping_callback(x, f, accept):
            print("x: %s f: %f accept: %d" % (str(x), f, accept))
            sys.stdout.flush()

        minimizer_kwargs = copy.deepcopy(self._minimizer_kwargs)
        minimizer_kwargs['bounds'] = bounds
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

        return yocomo.simulator.Predictor(self._simulator, **learned_params)
