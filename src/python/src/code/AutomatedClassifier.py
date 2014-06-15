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

data_directory=r'..\..\..\..\out\automated_cache'

features_run, experiment_name = EG.run_LSD_default_fq_given_interval(-0.25, 0.25)

cache_file_name = os.path.join(data_directory, experiment_name)
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
    logging.info('No cached version found. Calculated in %.3fs' % (time.clock() - t))

# convert classes to integers
classes_str = set(answers)
class_to_int_mapping = dict((c, i) for i, c in enumerate(classes_str))
answers = [class_to_int_mapping[class_str] for class_str in answers]

C_range = np.append(np.append(np.append(np.linspace(0.1, 1.0, 10), np.linspace(2.0, 10.0, 9)), np.linspace(20.0, 100.0, 9)), np.linspace(200.0, 1000.0, 9))
C_range = np.append(C_range, np.linspace(2000.0, 10000.0, 9))
C_range = np.append(C_range, np.linspace(20000.0, 100000.0, 9))
tc = time.clock()

skf = cross_validation.StratifiedKFold(answers, n_folds=5)
parameters = {'C': C_range}
gs_clf = grid_search.GridSearchCV(svm.SVC(), parameters, cv=skf)
gs_clf.fit(features_values, answers)
logging.info('Best parameters are: %s; With score: %.4f', str(gs_clf.best_params_), gs_clf.best_score_)

# max_score, max_C = 0., 0.
# for C in C_range:
#     classifier = svm.SVC(C=C, kernel='linear')
#
#     # scores = []
#     # for i in range(5):
#     #     X_train, X_test, y_train, y_test = cross_validation.train_test_split(features_values, answers, test_size=0.2, random_state=i)
#     #     classifier.fit(X_train, y_train)
#     #     score = classifier.score(X_test, y_test)
#     #     scores.append(score)
#
#     skf = cross_validation.StratifiedKFold(answers, n_folds=5)
#     # print('y vector: %s' % str(answers))
#     # for train, test in skf:
#     #     print("%s %s" % (train, test))
#     scores = cross_validation.cross_val_score(classifier, features_values, answers, cv=skf)
#     score = np.mean(scores)
#     if score > max_score:
#         max_score, max_C = score, C
#     logging.info('Cross validation scores for C=%.2f: [%s]. With mean: %.4f', C, ', '.join([str(score) for score in scores]), np.mean(scores))
#
#     # classifier.fit(features_values, answers)
#     # score = classifier.score(features_values, answers)
#     # logging.info('Got score %.4f%% on training data with C=%.3f', score, C)
# logging.info('Best score = %.4f on C = %.2f', max_score, max_C)
logging.info('Experiment have taken %.3fs', (time.clock() - tc))