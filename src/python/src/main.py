import logging
import time
from ECG_processor import runExperiment

logging.basicConfig(level=logging.DEBUG,
                    filename=u'..\\logs\\main.log',
                    format='[%(asctime)s] %(levelname)s: %(message)s {%(module)s.%(funcName)s:%(lineno)d}')

BYTES_FORMAT = [None]

filenames = [r'..\..\..\data\new_data\1_1.wav']

logging.info('===============')
logging.info('Program started')
logging.info('===============')

t = time.clock()
runExperiment(filenames, r'..\..\..\out\py_out.arff')
logging.info('Program ended in %.3fs' % (time.clock() - t))
