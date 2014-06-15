import datetime
import logging
import time
import numpy as np
import os

from src.code import ExperimentsGenerator as eg

FEATURES = {
    'RMS': 'features.RMS.RMS',
    'SpectralDensity': 'features.SpectralDensity.SpectralDensity',
    'LocalizedSpectralDensity': 'features.LocalizedSpectralDensity.LocalizedSpectralDensity'
}

# FEATURES_RUN = [
    # {'feature': 'RMS', 'name': 'RMS', 'options': {'normalized': True}},
    # {'feature': 'SpectralDensity', 'name': 'SpectralDensity1', 'options': {'normalized': True, 'begin': 25., 'end': 50., 'use_original_signal': True}},
    # {'feature': 'SpectralDensity', 'name': 'SpectralDensity2', 'options': {'normalized': True, 'begin': 50., 'end': 100., 'use_original_signal': True}},
    # {'feature': 'SpectralDensity', 'name': 'SpectralDensity3', 'options': {'normalized': True, 'begin': 100., 'end': 150., 'use_original_signal': True}},
    # {'feature': 'SpectralDensity', 'name': 'SpectralDensity4', 'options': {'normalized': True, 'begin': 150., 'end': 200., 'use_original_signal': True}},
    # {'feature': 'SpectralDensity', 'name': 'SpectralDensity5', 'options': {'normalized': True, 'begin': 200., 'end': 250., 'use_original_signal': True}},
    # {'feature': 'SpectralDensity', 'name': 'SpectralDensity6', 'options': {'normalized': True, 'begin': 250., 'end': 300., 'use_original_signal': True}},
    # {'feature': 'SpectralDensity', 'name': 'SpectralDensity7', 'options': {'normalized': True, 'begin': 300., 'end': 400., 'use_original_signal': True}},
    # {'feature': 'SpectralDensity', 'name': 'SpectralDensity8', 'options': {'normalized': True, 'begin': 400., 'end': 500., 'use_original_signal': True}},
    # {'feature': 'SpectralDensity', 'name': 'SpectralDensity9', 'options': {'normalized': True, 'begin': 500., 'end': 750., 'use_original_signal': True}},
    # {'feature': 'SpectralDensity', 'name': 'SpectralDensity10', 'options': {'normalized': True, 'begin': 750., 'end': 1000., 'use_original_signal': True}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity1_fq', 'options': {'normalized': True, 'beat_begin': -0.1, 'beat_end': 0.1,'fq_begin': 50., 'fq_end': 100., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity2_fq', 'options': {'normalized': True, 'beat_begin': -0.1, 'beat_end': 0.1,'fq_begin': 100., 'fq_end': 150., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity3_fq', 'options': {'normalized': True, 'beat_begin': -0.1, 'beat_end': 0.1,'fq_begin': 150., 'fq_end': 200., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity4_fq', 'options': {'normalized': True, 'beat_begin': -0.1, 'beat_end': 0.1,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity5_fq', 'options': {'normalized': True, 'beat_begin': -0.1, 'beat_end': 0.1,'fq_begin': 250., 'fq_end': 300., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity6_fq', 'options': {'normalized': True, 'beat_begin': -0.1, 'beat_end': 0.1,'fq_begin': 300., 'fq_end': 400., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity7_fq', 'options': {'normalized': True, 'beat_begin': -0.1, 'beat_end': 0.1,'fq_begin': 400., 'fq_end': 500., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity80_fq', 'options': {'normalized': True, 'beat_begin': -0.15, 'beat_end': 0.15,'fq_begin': 25., 'fq_end': 50., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity8_fq', 'options': {'normalized': True, 'beat_begin': -0.15, 'beat_end': 0.15,'fq_begin': 50., 'fq_end': 100., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity9_fq', 'options': {'normalized': True, 'beat_begin': -0.15, 'beat_end': 0.15,'fq_begin': 100., 'fq_end': 150., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity10_fq', 'options': {'normalized': True, 'beat_begin': -0.15, 'beat_end': 0.15,'fq_begin': 150., 'fq_end': 200., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity11_fq', 'options': {'normalized': True, 'beat_begin': -0.15, 'beat_end': 0.15,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity12_fq', 'options': {'normalized': True, 'beat_begin': -0.15, 'beat_end': 0.15,'fq_begin': 250., 'fq_end': 300., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity13_fq', 'options': {'normalized': True, 'beat_begin': -0.15, 'beat_end': 0.15,'fq_begin': 300., 'fq_end': 400., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity14_fq', 'options': {'normalized': True, 'beat_begin': -0.15, 'beat_end': 0.15,'fq_begin': 400., 'fq_end': 500., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity15_fq', 'options': {'normalized': True, 'beat_begin': -0.15, 'beat_end': 0.15,'fq_begin': 500., 'fq_end': 750., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity16_fq', 'options': {'normalized': True, 'beat_begin': -0.15, 'beat_end': 0.15,'fq_begin': 750., 'fq_end': 1000., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity1', 'options': {'normalized': True, 'beat_begin': 0.0, 'beat_end': 0.1,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity2', 'options': {'normalized': True, 'beat_begin': 0.1, 'beat_end': 0.2,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity3', 'options': {'normalized': True, 'beat_begin': 0.2, 'beat_end': 0.3,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity4', 'options': {'normalized': True, 'beat_begin': 0.3, 'beat_end': 0.4,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity5', 'options': {'normalized': True, 'beat_begin': 0.4, 'beat_end': 0.5,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity6', 'options': {'normalized': True, 'beat_begin': 0.5, 'beat_end': 0.6,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity7', 'options': {'normalized': True, 'beat_begin': 0.6, 'beat_end': 0.7,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity8', 'options': {'normalized': True, 'beat_begin': 0.7, 'beat_end': 0.8,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity9', 'options': {'normalized': True, 'beat_begin': 0.8, 'beat_end': 0.9,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity10', 'options': {'normalized': True, 'beat_begin': 0.9, 'beat_end': 1.0,'fq_begin': 200., 'fq_end': 250., 'calc_type': 'fft'}}
# ]

def importClass(className):
    nameSplit = className.split('.')
    module = '.'.join(nameSplit[0:-1])
    className = nameSplit[-1]
    mod = __import__(module, fromlist=[className])
    return getattr(mod,className)

def loadFeatures(features):
    featuresDict = {}
    for name, className in features.iteritems():
        feature = importClass(className)
        featuresDict[name] = feature
    return featuresDict

def updateRuns(featuresRun, featuresDict):
    for run in featuresRun:
        featureName = run['feature']
        run['feature'] = featuresDict[featureName]

def checkRuns(featuresRun):
    name_set = set()
    for run in featuresRun:
        run_name = run['name']
        assert run_name not in name_set, '"%s" name is not unique' % run_name
        name_set.add(run_name)

def runFeatures(ecg, runList):
    result = []
    for featureRun in runList:
        feature = featureRun['feature']()
        options = featureRun['options'] if ('options' in featureRun) else {}
        res = feature.run(ecg, **options)
        result.append(res)
    logging.info('Experiment result = %s' % str(result))
    return result

def load_files(description):
    assert 'loader' in description
    assert 'files' in description
    t = time.clock()

    loaderClass = importClass(description['loader'])
    files_list = description['files']
    objects = []
    for file_name in files_list:
        ecg = loaderClass(file_name)
        objects.append(ecg)
    time_to_load = (time.clock() - t)
    logging.info('Data loaded in %.3fs, avg %.3fs per object', time_to_load, time_to_load / float(len(objects)))
    return objects

def runExperiment(objects_to_evaluate, features_to_run, run_options,  outFilename=None):
    logging.info('Running experiment %s' % (str(features_to_run)))

    data = []
    target = []
    featuresDict = loadFeatures(FEATURES)
    featuresRun = features_to_run
    checkRuns(featuresRun)
    updateRuns(featuresRun, featuresDict)
    classesToRun = run_options['classes']

    outStr = []
    outStr.append('%% auto generated file from PYTHON on %s\n' % str(datetime.datetime.now()))
    outStr.append('@RELATION ecg_mining')
    for featureRun in featuresRun:
        outStr.append('@ATTRIBUTE %s %s' % (featureRun['name'], featureRun['feature'].type))
    outStr.append('@ATTRIBUTE class {%s}\n' % ', '.join(classesToRun))
    outStr.append('@DATA')

    counter = 0
    for ecg in objects_to_evaluate:
        counter+=1
        # if counter > 200:
        #     break
        logging.info('Processing %d/%d object [%s]' % (counter, len(objects_to_evaluate), ecg.name))
        tt = time.time()
        if classesToRun != 'all' and ecg.getClass() not in classesToRun:
            logging.info('Skipping file %s of class %s, because this class is not in scope', ecg.name, ecg.getClass())
            continue
        featuresValues = runFeatures(ecg, features_to_run)
        logging.info('%d file processed in %.3f seconds', counter, (time.time() - tt))
        data.append(featuresValues)
        target.append(ecg.getClass())
        outStr.append(', '.join(['%.5f' % v for v in featuresValues]) + ', ' + ecg.getClass())

    outStr = '\n'.join(outStr)
    if outFilename is not None:
        logging.info('Writing to [%s] data: %s' % (outFilename, outStr.replace('\n', r'\n')))
        with open(outFilename, 'w') as fo:
            fo.write(outStr)

    return data, target