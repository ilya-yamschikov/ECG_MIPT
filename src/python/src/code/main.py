import logging
import time
import os
import re

from src.code.ECG_processor import runExperiment


logging.basicConfig(level=logging.DEBUG,
                    filename=u'..\\..\\logs\\main.log',
                    format='[%(asctime)s] %(levelname)s: %(message)s {%(module)s.%(funcName)s:%(lineno)d}')
logging.getLogger().addHandler(logging.StreamHandler())


def generatePTBFilesList(descriptionFileName):
    filesList = []
    directory, _ = os.path.split(descriptionFileName)
    d = open(descriptionFileName, 'r')
    for fileName in d:
        filesList.append(os.path.join(directory, fileName.rstrip('\n')))
    return filesList

def generateMouseFilesList(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    pattern = r'^\d+_\d+\.wav$'
    files = [f for f in files if re.match(pattern, f)]
    resFilesList = [os.path.join(directory,f) for f in files]
    return resFilesList

data_mice = {
    'loader': 'ECG_loader.MouseECG',
    'files': generateMouseFilesList(r'..\..\..\..\data\new_data'),
    'classes': ['DO1', 'I10']
}
data_ptb = {
    'loader': 'ECG_loader.PTB_ECG',
    'files': generatePTBFilesList(r'..\..\..\..\data\ptb_database_csv\info.txt'),
    'classes': 'all'
}

logging.info('===============')
logging.info('Program started')
logging.info('===============')

t = time.clock()
runExperiment(data_mice, r'..\..\..\..\out\py_out.arff')
logging.info('Program ended in %.3fs' % (time.clock() - t))
