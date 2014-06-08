import datetime
import logging
import time
import numpy as np

FEATURES = {
    'RMS': 'features.RMS.RMS',
    'SpectralDensity': 'features.SpectralDensity.SpectralDensity',
    'LocalizedSpectralDensity': 'features.LocalizedSpectralDensity.LocalizedSpectralDensity'
}

FEATURES_RUN = [
    {'feature': 'RMS', 'name': 'RMS', 'options': {'normalized': True}},
    {'feature': 'SpectralDensity', 'name': 'SpectralDensity1', 'options': {'normalized': True, 'begin': 200., 'end': 250.}},
    {'feature': 'SpectralDensity', 'name': 'SpectralDensity2', 'options': {'normalized': True, 'begin': 250., 'end': 300.}},
    {'feature': 'SpectralDensity', 'name': 'SpectralDensity3', 'options': {'normalized': True, 'begin': 300., 'end': 400.}},
    {'feature': 'SpectralDensity', 'name': 'SpectralDensity4', 'options': {'normalized': True, 'begin': 400., 'end': 500.}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity1', 'options': {'normalized': True, 'beat_begin': 0., 'beat_end': 0.25,'fq_begin': 200., 'fq_end': 400.}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity2', 'options': {'normalized': True, 'beat_begin': 0.25, 'beat_end': 0.5,'fq_begin': 200., 'fq_end': 400.}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity3', 'options': {'normalized': True, 'beat_begin': 0.5, 'beat_end': 0.75,'fq_begin': 200., 'fq_end': 400.}},
    # {'feature': 'LocalizedSpectralDensity', 'name': 'LocalizedSpectralDensity4', 'options': {'normalized': True, 'beat_begin': 0.75, 'beat_end': 1.,'fq_begin': 200., 'fq_end': 400.}}
]

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

def runExperiment(data_description, outFilename):
    logging.info('Running experiment %s' % (str(FEATURES_RUN)))

    featuresDict = loadFeatures(FEATURES)
    featuresRun = FEATURES_RUN
    checkRuns(featuresRun)
    updateRuns(featuresRun, featuresDict)
    loaderClass = importClass(data_description['loader'])
    files = data_description['files']
    np.random.shuffle(files)

    outStr = []
    outStr.append('%% auto generated file from PYTHON on %s\n' % str(datetime.datetime.now()))
    outStr.append('@RELATION ecg_mining')
    for featureRun in featuresRun:
        outStr.append('@ATTRIBUTE %s %s' % (featureRun['name'], featureRun['feature'].type))
    outStr.append('@ATTRIBUTE class {HEALTHY, MI}\n')
    outStr.append('@DATA')

    counter = 0
    for ecgFileName in files:
        counter+=1
        if counter > 200:
            break
        logging.info('Processing %d/%d file [%s]' % (counter, len(files), ecgFileName))
        tt = time.time()
        ecg = loaderClass(ecgFileName)
        featuresValues = runFeatures(ecg, FEATURES_RUN)
        logging.info('%d file processed in %.3f seconds', counter, (time.time() - tt))
        outStr.append(', '.join(['%.5f' % v for v in featuresValues]) + ', ' + ecg.getClass())

    outStr = '\n'.join(outStr)
    logging.info('Writing to [%s] data: %s' % (outFilename, outStr.replace('\n', r'\n')))
    fo = open(outFilename, 'w')
    fo.write(outStr)
    fo.close()