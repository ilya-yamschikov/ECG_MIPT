import logging
import time
import os

import src.code.ECG_processor as processor
from src.code import ExperimentsGenerator as EG
from src.code import config


logging.basicConfig(level=logging.DEBUG,
                    filename=u'..\\..\\logs\\main.log',
                    format='[%(asctime)s] %(levelname)s: %(message)s {%(module)s.%(funcName)s:%(lineno)d}')
logging.getLogger().addHandler(logging.StreamHandler())

logging.info('===============')
logging.info('Program started')
logging.info('===============')


data = config.data_mice
features_run, experiment_name = EG.run_LSD_default_fq_given_interval(-0.25, 0.25)
objects_to_evaluate = processor.load_files(data)

t = time.clock()
processor.runExperiment(objects_to_evaluate, features_run, data['options'], outFilename=os.path.join(r'..\..\..\..\out', experiment_name))
logging.info('Program ended in %.3fs' % (time.clock() - t))
