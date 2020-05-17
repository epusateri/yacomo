from yacomo.util import log_error, log_warn, log_info, log_verbose, is_debug


class Objective:
    pass

class ObjAmethyst:

    def __init__(self, config, simulator):
        self._scale = config['scale']        
        self._simulator = simulator

    def set_target_df(self, target_df):
        self._target_df = target_df

    def compute(self,
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
        self._simulator.set_parameters(r0_before,
                                       day_sd_start,
                                       r0_after,
                                       day_goner_0,
                                       sigmoid_param)
        log_debug(self._simulator)
        log_debug(self._simulator._r0_before)
        log_debug(self._simulator._r0_after)
        log_debug(self._simulator._day_goner_0)
        
        estimate_df = self._simulator.run(len(self._target_df))

        log_debug(self._simulator._day_goner_0)

        mse = ((estimate_df - self._target_df)**2).sum()/len(self._target_df)
        mse *= self._scale

        return mse
