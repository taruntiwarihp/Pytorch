from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from logging.handlers import TimedRotatingFileHandler
import os

def create_logger(log_dir='logs'):
    
    logger = logging.getLogger('ModularFace')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logName = os.path.join(log_dir, 'flaskApp.log')
    logHandler = TimedRotatingFileHandler(logName, when='D', interval=1, backupCount=0)
    logHandler.setLevel(logging.INFO)
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger