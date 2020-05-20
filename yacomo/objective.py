import time

from yacomo.util import log_error, log_warn, log_info, log_verbose, log_debug, is_debug


class Objective:
    pass

class ObjAmethyst:

    def __init__(self, config, simulator):
        self._scale = config['scale']        
        self._simulator = simulator

        self._compute_count = 0
        self._simulator_time = 0.0
        self._compute_time = 0.0

    def set_target_df(self, target_df):
        self._target_df = target_df

    def compute(self,
                r0_before,
                day_sd_start,
                r0_after,
                day_goner_0,
                sigmoid_param):
        
        self._simulator.set_parameters(r0_before,
                                       day_sd_start,
                                       r0_after,
                                       day_goner_0,
                                       sigmoid_param)
        compute_before = time.monotonic()

        before = time.monotonic()
        estimate_df = self._simulator.run(len(self._target_df))
        self._simulator_time += time.monotonic() - before

        mse = ((estimate_df - self._target_df)**2).sum()/len(self._target_df)
        mse *= self._scale

        self._compute_time += time.monotonic() - compute_before
        
        return mse
