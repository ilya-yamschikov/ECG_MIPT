import os
import time
import logging
import arff
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import svm, cross_validation, grid_search
from sklearn.metrics import roc_curve, auc

import src.code.ECG_processor as processor
from src.code import ExperimentsGenerator as EG
from src.code import config

logging.basicConfig(level=logging.DEBUG,
                    filename=u'..\\..\\logs\\automated.log',
                    format='[%(asctime)s] %(levelname)s: %(message)s {%(module)s.%(funcName)s:%(lineno)d}')
logging.getLogger().addHandler(logging.StreamHandler())

CACHE_DIRECTORY=r'..\..\..\..\out\automated_cache'

C_RANGE = np.append(np.append(np.append(np.append(np.linspace(0.01, 0.09, 10), np.linspace(0.1, 1.0, 10)), np.linspace(2.0, 10.0, 9)), np.linspace(20.0, 100.0, 9)), np.linspace(200.0, 1000.0, 9))
C_RANGE = np.append(C_RANGE, np.linspace(2000.0, 10000.0, 9))
C_RANGE = np.append(C_RANGE, np.linspace(20000.0, 100000.0, 9))

# do not load ECGs from files again
OBJECTS_LOADED = [None]

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
        if OBJECTS_LOADED[0] is None:
            objects_to_evaluate = processor.load_files(data)
            OBJECTS_LOADED[0] = objects_to_evaluate
        features_values, answers = processor.runExperiment(OBJECTS_LOADED[0], features_run, data['options'], outFilename=cache_file_name)
        logging.info('No cached version found. Calculated new version in %.3fs' % (time.clock() - t))

    # convert classes to integers
    classes_str = set(answers)
    class_to_int_mapping = dict((c, i) for i, c in enumerate(classes_str))
    answers = [class_to_int_mapping[class_str] for class_str in answers]
    return features_values, answers

def randomize_sampling(features_values, answers):
    idxs = np.arange(len(answers))
    np.random.shuffle(idxs)
    features_values = [features_values[i] for i in idxs]
    answers = [answers[i] for i in idxs]
    return features_values, answers

# tc = time.clock()
# search_range = 0.4
# # centers = np.linspace(0., 0.975, 40)
# centers=[0.0]
# centers_scores = {}
# fq_ranges_count = 5
# cv_repeats = 10
# for center in centers:
#     tc_inner = time.clock()
#     features_run, experiment_name = EG.run_LSD_given_fq_ranges_count_and_given_interval(center - search_range / 2., center + search_range / 2., fq_ranges_count)
#     # features_run, experiment_name = EG.run_SD_given_fq_ranges_count(fq_ranges_count, fq_begin=25., fq_end=1000.)
#     features_values, answers = get_experiment_data(features_run, experiment_name)
#
#     cv_scores = []
#     for i in range(cv_repeats):
#         shuffled_values, shuffled_answers = randomize_sampling(features_values, answers)
#         skf = cross_validation.StratifiedKFold(shuffled_answers, n_folds=5)
#         gs_clf = grid_search.GridSearchCV(svm.SVC(kernel='linear'), {'C': C_RANGE}, cv=skf)
#         gs_clf.fit(shuffled_values, shuffled_answers)
#         cv_scores.append(gs_clf.best_score_)
#         logging.info('Best parameters are: %s; With score: %.4f; Center in %.4f; Calc in %.3fs', str(gs_clf.best_params_), gs_clf.best_score_, center, (time.clock() - tc_inner))
#     logging.info('%d times CV score: %f; Stdev: %f', cv_repeats, np.mean(cv_scores), np.std(cv_scores))
#     centers_scores[center] = np.mean(cv_scores)
#
# logging.info('Experiment have taken %.3fs', (time.clock() - tc))
# logging.info('Scores for centers [%s]:\n[%s]', ', '.join([str(center) for center in centers]), ', '.join(['%.4f' % centers_scores[center] for center in centers]))


# ROC curve
fq_ranges_count = 5
cv_repeats = 1
# features_run, experiment_name = EG.run_SD_given_fq_ranges_count(fq_ranges_count, fq_begin=25., fq_end=1000.)
center = 0.0
search_range = 0.4
features_run, experiment_name = EG.run_LSD_given_fq_ranges_count_and_given_interval(center - search_range / 2., center + search_range / 2., fq_ranges_count)
features_values, answers = get_experiment_data(features_run, experiment_name)

# get best C:
skf = cross_validation.StratifiedKFold(answers, n_folds=5)
gs_clf = grid_search.GridSearchCV(svm.SVC(kernel='linear'), {'C': C_RANGE}, cv=skf)
gs_clf.fit(features_values, answers)
right_C = gs_clf.best_params_['C']
logging.info('Best C: %f; Classification: %f', right_C, gs_clf.best_score_)

all_tpr = []
all_fpr = []
for i in range(cv_repeats):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    shuffled_values, shuffled_answers = randomize_sampling(features_values, answers)
    shuffled_values = np.array(shuffled_values)
    shuffled_answers = np.array(shuffled_answers)
    skf = cross_validation.StratifiedKFold(shuffled_answers, n_folds=5)
    classifier = svm.SVC(kernel='linear', C=right_C, probability=True)
    for (train, test) in skf:
        probas_ = classifier.fit(shuffled_values[train], shuffled_answers[train]).predict_proba(shuffled_values[test])
        fpr, tpr, thresholds = roc_curve(shuffled_answers[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

    mean_tpr /= len(skf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    all_tpr.append(mean_tpr)
    all_fpr.append(mean_fpr)
    plt.plot(mean_fpr, mean_tpr, 'k-')
    logging.info('Plot: x = [%s]; y = [%s]', ', '.join(['%.4f' % fpr for fpr in  mean_fpr]), ', '.join(['%.6f' % tpr for tpr in  mean_tpr]))
#
# tot_tpr = 0.0
# tot_fpr = np.linspace(0, 1, 100)
# for tpr, fpr in zip(all_tpr, all_fpr):
#     tot_tpr += interp(tot_fpr, fpr, tpr)
#     tot_tpr[0] = 0.
# tot_tpr /= len(tot_tpr)
# tot_tpr[-1] = 1.0
# plt.plot(tot_fpr, tot_tpr, 'k--')
plt.show()