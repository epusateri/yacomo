import logging
import copy
import numpy as np

from yacomo.util import log_error, log_warn, log_debug, log_verbose, log_info, is_debug


class Simulator:
    pass

class Predictor:    
    def __init__(self, simulator, **kwargs):
        self._simulator = copy.deepcopy(simulator)
        self._simulator.set_parameters(**kwargs)

    @staticmethod
    def from_parameters(parameters):
        return Predictor(SimAntelope(parameters['fixed']), **parameters['learned'])

    def run(self, n_days):
        return self._simulator.run(n_days)

    def parameters(self):
        return self._simulator.parameters()

class SimAntelope:

    def __init__(self, config):
        self._smoothing_window_size = config['smoothing_window_size']
        self._days_to_death = config['days_to_death']
        self._days_contagious = config['days_contagious']

    def set_parameters(self,
                       r0_before,
                       day_sd_start,
                       r0_after,
                       day_first_goner,
                       sigmoid_param):

        # Parameters to model R0
        self._r0_before = r0_before
        self._day_sd_start = day_sd_start
        self._r0_after = r0_after
        self._sigmoid_param = sigmoid_param

        # Other simulator parameters
        self._day_first_goner = day_first_goner + 0.001

    def parameters(self):
        params = {
            'fixed': {
                'smoothing_window_size': self._smoothing_window_size,
                'days_to_death': self._days_to_death,
                'days_contagious': self._days_contagious
            },
            'learned': {
                'r0_before': self._r0_before,
                'day_sd_start': self._day_sd_start,
                'r0_after': self._r0_after,
                'sigmoid_param': self._sigmoid_param,
                'day_first_goner': self._day_first_goner
            }
        }
        return params
        
    def _compute_r0(self, n_days):
        day = np.arange(-self._day_sd_start, n_days - self._day_sd_start)
        # log_verbose('day: %s' % (day))
        # log_verbose('day_sd_start: %d' % (self._day_sd_start))
        # log_verbose('n_days: %d' % (n_days))
        sigmoid = -1/(1 + np.exp(-self._sigmoid_param * day)) * (self._r0_before - self._r0_after) + self._r0_before
        r0 = sigmoid
        return r0
    
    def run(self, n_days):
        r0 = self._compute_r0(n_days)

        daily_goners = [0.0]*n_days
        
        # Smooth initial goner day        
        first_goner_day = self._day_first_goner - self._days_contagious/2.0 + 0.01
        for delta in range(0, self._days_contagious):
            day = first_goner_day + delta
            
            day_floor = int(np.floor(day))
            day_ceil = int(np.ceil(day))

            right_frac = day - day_floor
            left_frac = day_ceil - day

            daily_goners[day_floor] += left_frac/self._days_contagious
            daily_goners[day_ceil] += right_frac/self._days_contagious
            
        # Run simulation
        for day in range(0, n_days):
            for c in range(1, self._days_contagious + 1):
                if day + c >= n_days:
                    break
                daily_goners[day + c] += r0[day]*daily_goners[day]/self._days_contagious

        # TODO: Fencepost error?
        daily_deaths = [0.0]*(self._days_to_death + self._smoothing_window_size)
        daily_deaths.extend(daily_goners)
        daily_deaths = daily_deaths[:n_days]

        return daily_deaths

