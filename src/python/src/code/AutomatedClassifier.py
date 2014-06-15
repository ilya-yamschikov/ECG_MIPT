import os
import time
import logging
import arff
import copy
import numpy as np
from sklearn import svm, cross_validation, grid_search

import src.code.ECG_processor as processor
from src.code import ExperimentsGenerator as EG
from src.code import config

logging.basicConfig(level=logging.DEBUG,
                    filename=u'..\\..\\logs\\automated.log',
                    format='[%(asctime)s] %(levelname)s: %(message)s {%(module)s.%(funcName)s:%(lineno)d}')
logging.getLogger().addHandler(logging.StreamHandler())

CACHE_DIRECTORY=r'..\..\..\..\out\automated_cache'

C_RANGE = np.append(np.append(np.append(np.linspace(0.1, 1.0, 10), np.linspace(2.0, 10.0, 9)), np.linspace(20.0, 100.0, 9)), np.linspace(200.0, 1000.0, 9))
C_RANGE = np.append(C_RANGE, np.linspace(2000.0, 10000.0, 9))
C_RANGE = np.append(C_RANGE, np.linspace(20000.0, 100000.0, 9))

def get_experiment_data(features_run, experiment_name):
    cache_file_name = os.path.join(CACHE_DIRECTORY, experiment_name)

    if os.path.isfile(cache_file_name):
        # cached version exisits
        t = time.clock()
        with open(cache_file_name, 'r') as f:
            raw_file_data = arff.load(f)
        classIdx = [i for i, attr in enumerate(raw_file_data[u'attributes']) if attr[0] == u'class']
        assert len(classIdx) == 1
        classIdx = classIdx[0]
        raw_data = raw_file_data[u'data']
        np.random.shuffle(raw_data)
        answers = [values[classIdx] for values in raw_data]
        features_values = copy.deepcopy(raw_data)
        for object_values in features_values:
            del object_values[classIdx]
        logging.info('Cached version loaded in %.3fs' % (time.clock() - t))
    else:
        t = time.clock()
        data = config.data_mice
        objects_to_evaluate = processor.load_files(data)
        features_values, answers = processor.runExperiment(objects_to_evaluate, features_run, data['options'], outFilename=cache_file_name)
        logging.info('No cached version found. Calculated new version in %.3fs' % (time.clock() - t))

    # convert classes to integers
    classes_str = set(answers)
    class_to_int_mapping = dict((c, i) for i, c in enumerate(classes_str))
    answers = [class_to_int_mapping[class_str] for class_str in answers]
    return features_values, answers

tc = time.clock()
search_range = 0.5
centers = np.linspace(0., 0.9, 10)
fq_ranges_count = 8
for center in centers:
    tc_inner = time.clock()
    features_run, experiment_name = EG.run_LSD_given_fq_ranges_count_and_given_interval(center - search_range / 2., center + search_range / 2., fq_ranges_count)
    features_values, answers = get_experiment_data(features_run, experiment_name)

    skf = cross_validation.StratifiedKFold(answers, n_folds=5)
    gs_clf = grid_search.GridSearchCV(svm.SVC(kernel='linear'), {'C': C_RANGE}, cv=skf)
    gs_clf.fit(features_values, answers)
    logging.info('Best parameters are: %s; With score: %.4f; Calc in %.3fs', str(gs_clf.best_params_), gs_clf.best_score_, (time.clock() - tc_inner))
logging.info('Experiment have taken %.3fs', (time.clock() - tc))