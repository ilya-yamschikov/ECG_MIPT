import logging
import time
import os
import re

import src.code.ECG_processor as processor
from src.code import ExperimentsGenerator as EG


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
    'options': {'classes': ['DO1', 'I10']}
}
data_ptb = {
    'loader': 'ECG_loader.PTB_ECG',
    'files': generatePTBFilesList(r'..\..\..\..\data\ptb_database_csv\info.txt'),
    'options': {'classes': 'all'}
}

logging.info('===============')
logging.info('Program started')
logging.info('===============')


data = data_mice
features_run, experiment_name = EG.run_LSD_default_fq_given_interval(-0.25, 0.25)
objects_to_evaluate = processor.load_files(data)

t = time.clock()
processor.runExperiment(objects_to_evaluate, features_run, data['options'], outFilename=os.path.join(r'..\..\..\..\out', experiment_name))
logging.info('Program ended in %.3fs' % (time.clock() - t))
