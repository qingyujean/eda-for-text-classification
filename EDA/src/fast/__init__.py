#coding=utf-8

import logging
import time
import fasttext
import pkg_resources
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, classification_report

# 构建fasttext需要的输入数据的格式
def create_faststyle_inputdata(filepath, cols_index):
    df = pd.read_csv(filepath, encoding='UTF-8', sep='\t', header=0, index_col=False, usecols=cols_index)
    df = df.dropna()

    faststyle_label = ['__label__' + str(cls) for cls in df['label']]
    df['label'] = faststyle_label

    return df


def train_fasttext_model(traindata_file, params=None):
    starttime = time.time()
    logging.info('开始训练fasttext模型...')

    fast_model = fasttext.train_supervised(traindata_file,
                                           label_prefix='__label__',
                                           thread=3,
                                           **params)

    endtime = time.time()
    logging.info('训练fasttext模型耗时：%.2f s' % (endtime - starttime,))
    return fast_model


def save_fast_model(fast, model_path):  # 并保存模型
    # 年月日_时分秒
    cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))  # 给保存的模型的名字加上时间标签，以区分训练过程中产生的不同的模型
    model_name = 'fast' + '_' + cur_time + '.pkl'
    logging.info('fasttext模型名称：%s' % model_name)
    save_model_file = model_path + model_name
    # 保存模型
    fast.save_model(save_model_file)


def load_fast_model(model_path):
    # 加载模型
    starttime = time.time()
    fast = fasttext.load_model(model_path)
    endtime = time.time()
    logging.info('加载fasttext模型耗时：%.2f s' % (endtime - starttime,))
    return fast


def fast_predict(fast_model, content_seg_list, k=1):
    starttime = time.time()
    predicted = fast_model.predict(content_seg_list, k=7)########如果k>=2，概率是按从大到小的顺序排列的
    y_pred = [int(label[0][-1:]) for label in predicted[0] ]#label[0]  __label__0
    y_pred_prob = []
    for i in range(len(predicted[1])):
        y_pred_prob.append([predicted[1][i][0]])

    endtime = time.time()
    logging.info('预测耗时：%.2f s' % (endtime-starttime,))
    return y_pred, y_pred_prob


#计算分类评价指标
def get_metrics(y_true, y_pred, n_classes):
    metrics = {}

    if n_classes==2:
        #二分类
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=1)
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=1)
        metrics['f1'] = f1_score(y_true, y_pred, pos_label=1)
    else:#多分类
        average = 'macro'
        metrics[average+'_precision'] = precision_score(y_true, y_pred, average=average)
        metrics[average+'_recall'] = recall_score(y_true, y_pred, average=average)
        metrics[average+'_f1'] = f1_score(y_true, y_pred, average=average)
    

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    metrics['classification_report'] = classification_report(y_true, y_pred)
    
    return metrics