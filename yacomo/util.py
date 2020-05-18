import logging
import inspect
import os

_VERBOSE_LOG_LEVEL = logging.DEBUG + 1
assert(_VERBOSE_LOG_LEVEL < logging.INFO)

def init_logging(debug, verbose):
    
    logging.basicConfig(format = '%(asctime)s :: %(levelname)7s :: %(message)s')
    logging.addLevelName(_VERBOSE_LOG_LEVEL, 'VERBOSE')
    
    logging.getLogger().setLevel(logging.INFO)
    if verbose:
        logging.getLogger().setLevel(_VERBOSE_LOG_LEVEL)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
def is_debug():
    return (logging.getLogger().getLevel() == logging.DEBUG)
    
def _enrich_msg(msg):
    frame_info = inspect.stack()[2]
    bn = os.path.basename(frame_info.filename)
    ln = frame_info.lineno
    enriched = '{bn}:{ln} :: {msg}'.format(bn = bn, ln = ln, msg = msg)
    return enriched
    
def log_error(msg, *args, **kwargs):
    logging.error(_enrich_msg(msg), *args,**kwargs)
        
def log_warn(msg, *args, **kwargs):
    logging.warn(_enrich_msg(msg), *args,**kwargs)

def log_info(msg, *args, **kwargs):
    logging.info(_enrich_msg(msg), *args, **kwargs)

def log_verbose(msg, *args, **kwargs):
    logging.log(_VERBOSE_LOG_LEVEL, _enrich_msg(msg), *args,**kwargs)

def log_debug(msg, *args, **kwargs):
    logging.debug(_enrich_msg(msg), *args,**kwargs)
