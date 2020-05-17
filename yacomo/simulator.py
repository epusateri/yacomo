import logging
import copy
import numpy as np

from yacomo.util import log_error, log_warn, log_debug, log_verbose, log_info, is_debug


class Simulator:
    pass

class Predictor:    
    def __init__(self, simulator, **kwargs):
        self._simulator = simulator
        self._simulator.set_parameters(**kwargs)

    def run(self, n_days):
        self._simulator(n_days)

    def save(self):
        pass

class SimAntelope:

    def __init__(self, config):
        self._smoothing_window_size = config['smoothing_window_size']
        self._days_to_death = config['days_to_death']
        self._days_contagious = config['days_contagious']

    def set_parameters(self,
                       r0_before,
                       day_sd_start,
                       r0_after,
                       day_goner_0,
                       sigmoid_param):

        log_debug('%f %f %f %f %f',
                      r0_before,
                      day_sd_start,
                      r0_after,
                      sigmoid_param,
                      day_goner_0)
        
        # Parameters to model R0
        self._r0_before = r0_before
        self._day_sd_start = day_sd_start
        self._r0_after = r0_after
        self._sigmoid_param = sigmoid_param

        # Other simulator parameters
        self._day_goner_0 = day_goner_0 + 0.001

        log_debug('r0_after: %f', self._r0_after)
        log_debug('day_goner_0: %f', self._day_goner_0)

    def _compute_r0(self, n_days):
        day = np.arange(-self._day_sd_start, n_days - self._day_sd_start)
        sigmoid = -1/(1 + np.exp(-self._sigmoid_param * day)) * (self._r0_before - self._r0_after) + self._r0_before
        r0 = sigmoid
        return r0
    
    def run(self, n_days):

        log_debug(self)
        log_debug('r0_before: %f', self._r0_before)
        log_debug('r0_after: %f', self._r0_after)
        log_debug('day_goner_0: %d', self._day_goner_0)

        r0 = self._compute_r0(n_days)

        log_debug('daily_goners: %d', n_days)
        daily_goners = [0.0]*n_days
        
        # Smooth initial goner day
        first_goner_day = self._day_goner_0 - self._days_contagious/2.0
        for delta in range(0, self._days_contagious):
            log_debug('delta: %d', delta)
            day = first_goner_day + delta

            day_floor = int(np.floor(day))
            day_ceil = int(np.ceil(day))

            left_frac = day - day_floor
            right_frac = day_ceil - day

            log_debug('day_ceil: %d day_floor: %d', day_ceil, day_floor)
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

