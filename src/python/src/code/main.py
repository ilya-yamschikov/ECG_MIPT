import logging
import time
from os.path import split, join

from src.code.ECG_processor import runExperiment


logging.basicConfig(level=logging.DEBUG,
                    filename=u'..\\..\\logs\\main.log',
                    format='[%(asctime)s] %(levelname)s: %(message)s {%(module)s.%(funcName)s:%(lineno)d}')
logging.getLogger().addHandler(logging.StreamHandler())


def generatePTBFilesList(descriptionFileName):
    filesList = []
    directory, _ = split(descriptionFileName)
    d = open(descriptionFileName, 'r')
    for fileName in d:
        filesList.append(join(directory, fileName.rstrip('\n')))
    return filesList

data_mice = {
    'loader': 'ECG_loader.MouseECG',
    'files': [r'..\..\..\data\new_data\1_1.wav']
}
data_ptb = {
    'loader': 'ECG_loader.PTB_ECG',
    'files': generatePTBFilesList(r'..\..\..\..\data\ptb_database_csv\info.txt')
}

logging.info('===============')
logging.info('Program started')
logging.info('===============')

t = time.clock()
runExperiment(data_ptb, r'..\..\..\..\out\py_out.arff')
logging.info('Program ended in %.3fs' % (time.clock() - t))
