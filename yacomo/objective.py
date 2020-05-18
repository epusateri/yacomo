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
        
        # self._compute_count += 1
        # if self._compute_count % 100 == 0:
        #     log_verbose("compute_count: %d", self._compute_count)
        #     log_verbose("simulator_time: %f", self._simulator_time)
        #     log_verbose("compute_time: %f", self._compute_time)
        #     self._compute_time = 0.0
        #     self._simulator_time = 0.0
            
        # log_debug('%f %f %f %f %f',
        #               r0_before,
        #               day_sd_start,
        #               r0_after,
        #               sigmoid_param,
        #               day_goner_0)
        self._simulator.set_parameters(r0_before,
                                       day_sd_start,
                                       r0_after,
                                       day_goner_0,
                                       sigmoid_param)
        compute_before = time.monotonic()
        # log_debug(self._simulator)
        # log_debug(self._simulator._r0_before)
        # log_debug(self._simulator._r0_after)
        # log_debug(self._simulator._day_goner_0)

        before = time.monotonic()
        estimate_df = self._simulator.run(len(self._target_df))
        self._simulator_time += time.monotonic() - before

        # log_debug(self._simulator._day_goner_0)

        mse = ((estimate_df - self._target_df)**2).sum()/len(self._target_df)
        mse *= self._scale

        self._compute_time += time.monotonic() - compute_before
        
        return mse
