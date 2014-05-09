import datetime
import logging
from ECG_loader import MouseECG

FEATURES = {
    'RMS': 'features.RMS.RMS',
    'SpectralDensity': 'features.SpectralDensity.SpectralDensity'
}

FEATURES_RUN = [
    {'feature': 'RMS', 'name': 'RMS', 'options': {'normalized': True}},
    {'feature': 'SpectralDensity', 'name': 'SpectralDensity1', 'options': {'normalized': True, 'begin': 200., 'end': 250.}},
    {'feature': 'SpectralDensity', 'name': 'SpectralDensity2', 'options': {'normalized': True, 'begin': 250., 'end': 300.}},
    {'feature': 'SpectralDensity', 'name': 'SpectralDensity3', 'options': {'normalized': True, 'begin': 300., 'end': 400.}},
    {'feature': 'SpectralDensity', 'name': 'SpectralDensity4', 'options': {'normalized': True, 'begin': 400., 'end': 500.}}
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
    updateRuns(featuresRun, featuresDict)
    loaderClass = importClass(data_description['loader'])
    files = data_description['files']

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
        logging.info('Processing %d/%d file [%s]' % (counter, len(files), ecgFileName))
        ecg = loaderClass(ecgFileName)
        featuresValues = runFeatures(ecg, FEATURES_RUN)
        outStr.append(', '.join(['%.5f' % v for v in featuresValues]) + ', ' + ecg.getClass())

    outStr = '\n'.join(outStr)
    logging.info('Writing to [%s] data: %s' % (outFilename, outStr.replace('\n', r'\n')))
    fo = open(outFilename, 'w')
    fo.write(outStr)
    fo.close()