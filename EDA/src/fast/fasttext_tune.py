# coding=utf-8
"""
Author  : Jane
Contact : xijian@ict.ac.cn
Time    : 2020/9/29 9:28
Desc:
"""
import platform
import fasttext
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import copy
import tempfile
import shutil

if platform.system().lower() == 'windows':
    DATA_PATH = 'E:/jsnews_extra/data/'
else:
    DATA_PATH = '/home/xijian/jsnews_extra/data/'

"""
tuned_parameters = {
    'lr': [0.1, 0.05],
    'epoch': [15, 20, 25, 30],
    'dim': [50, 100, 150, 200],
    'wordNgrams': [2, 3],
}
"""
"""
best_score 0.8270148159514432
best_params {'lr': 0.1, 'epoch': 30, 'dim': 50, 'wordNgrams': 2}
"""

tuned_parameters = {
    'lr': [0.1],
    'epoch': [25, 30, 35],
    'dim': [10,30,50],
    'wordNgrams': [2],
}

"""
best_score 0.9409697713671934
best_params {'lr': 0.1, 'epoch': 35, 'dim': 10, 'wordNgrams': 2}
"""

# 计算分类评价指标
def get_metrics(y_true, y_pred, n_classes):
    metrics = {}
    if n_classes == 2:
        # 二分类
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=1)
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=1)
        metrics['f1'] = f1_score(y_true, y_pred, pos_label=1)
    else:  # 多分类
        average = 'macro'  # 'weighted'
        metrics['precision'] = precision_score(y_true, y_pred, average=average)
        metrics['recall'] = recall_score(y_true, y_pred, average=average)
        metrics['f1'] = f1_score(y_true, y_pred, average=average)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    return metrics


def get_gridsearch_params(param_grid):
    params_combination = [dict()]  # 所有可能的参数组合
    for k, v_list in param_grid.items():
        tmp = [{k: v} for v in v_list]
        # print('tmp:', tmp)
        n = len(params_combination)
        # params_combination = params_combination*len(tmp)  # 浅拷贝，有问题
        copy_params = [copy.deepcopy(params_combination) for _ in range(len(tmp))]  # params_combination(tmp)遍
        params_combination = sum(copy_params, [])
        _ = [params_combination[i*n+k].update(tmp[i]) for k in range(n) for i in range(len(tmp))]
    return params_combination


def get_KFold_scores(df, params, kf, metric, n_classes):
    metric_score = 0.0

    for train_idx, val_idx in kf.split(df):
        # print('train_idx:', train_idx, 'test_idx:', val_idx)
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        # print(df_train.head())

        tmpdir = tempfile.mkdtemp()
        # print(tmpdir)
        tmp_train_file = tmpdir + '/train.txt'
        df_train.to_csv(tmp_train_file, sep='\t', index=False, header=None, encoding='UTF-8')  # 不要表头

        fast_model = fasttext.train_supervised(tmp_train_file, label_prefix='__label__', thread=3, **params)
        # print(df_val[0])
        predicted = fast_model.predict(df_val[0].tolist())  # ([label...], [probs...])
        y_val_pred = [int(label[0][-1:]) for label in predicted[0]]  # label[0]  __label__0
        y_val = [int(cls[-1:]) for cls in df_val[1]]
        # print(y_val_pred)
        # print(y_val)
        score = get_metrics(y_val, y_val_pred, n_classes)[metric]
        metric_score += score
        shutil.rmtree(tmpdir, ignore_errors=True)
        """
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        # print(y_val_pred)
        score = get_metrics(y_val, y_val_pred, n_classes)[metric]
        # print(score)
        metric_score += score
        """
    print('平均分:', metric_score / kf.n_splits)
    return metric_score / kf.n_splits


def my_gridsearch_cv(df, param_grid, metrics, kfold=10):
    n_classes = len(np.unique(df[1]))
    print('n_classes', n_classes)

    kf = KFold(n_splits=kfold)  # k折交叉验证

    params_combination = get_gridsearch_params(param_grid)

    best_score = 0.0
    best_params = dict()
    for params in params_combination:
        print(params)
        # check_model = copy.deepcopy(model)
        # check_model.set_params(**params)
        # print(check_model.get_params())


        avg_score = get_KFold_scores(df, params, kf, metrics, n_classes)
        if avg_score > best_score:
            best_score = avg_score
            best_params = copy.deepcopy(params)

    # print('best_params:', best_params)
    return best_score, best_params


# my_gridsearch_cv(svc_model, X, y, tuned_parameters, '', kfold=5)

if __name__ == '__main__':
    filepath = DATA_PATH + 'js_news/labeled_data/20200927/fast/augmented/js_pd_tagged_train.txt'
    df = pd.read_csv(filepath, encoding='UTF-8', sep='\t', header=None, index_col=False, usecols=[0, 1])
    print(df.head())
    print(df.shape)  # (1710, 2)
    best_score, best_params = my_gridsearch_cv(df, tuned_parameters, 'accuracy', kfold=5)
    print('best_score', best_score)
    print('best_params', best_params)

