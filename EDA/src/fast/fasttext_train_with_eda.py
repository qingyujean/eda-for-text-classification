# coding=utf-8
'''
Created on 2020-4-20

@author: Yoga
'''
import logging
import pkg_resources
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fast import create_faststyle_inputdata, train_fasttext_model, \
    fast_predict, get_metrics, save_fast_model

DATA_PATH = '../data/'

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # 1.处理训练数据，变成fasttext认识的格式
    # 增强后的数据 content_seg_augment	label
    df_train = create_faststyle_inputdata(DATA_PATH + 'augmented/js_pd_tagged_train.txt', [0, 1])
    df_train.to_csv(DATA_PATH+'fast/augmented/js_pd_tagged_train.txt', sep='\t', index=False, header=None, encoding='UTF-8')  # 不要表头

    # 2.训练fasttext模型并保存
    best_params = {'lr': 0.1, 'epoch': 30, 'dim': 10, 'wordNgrams': 2}
    fast_model = train_fasttext_model(
        DATA_PATH + 'fast/augmented/js_pd_tagged_train.txt',  # eda增强后的数据
        params=best_params)

    # 训练接上的效果
    df_train_fast = pd.read_csv(DATA_PATH + 'fast/augmented/js_pd_tagged_train.txt', header=None,
                                encoding='UTF-8', sep='\t', index_col=False)  # eda增强后的数据
    df_train_fast[1].value_counts().plot(kind='barh') # 增强后的数据分布
    plt.show()

    X_train = [str(x).strip() for x in df_train_fast[0]]
    y_train = [int(cls[-1:]) for cls in df_train_fast[1]]
    y_pred, y_pred_prob = fast_predict(fast_model, X_train, k=1)  #参数 k=1为默认值

    #计算评价指标
    n_classes = len(np.unique(y_train))
    eval_metrics = get_metrics(y_train, y_pred, n_classes)
    logging.info('*'*15 + ' eval results ' + '*'*15)
    for key in sorted(eval_metrics.keys()):
        if key in ('confusion_matrix', 'classification_report'):
            logging.info("  %s = \n%s", key, str(eval_metrics[key]))
        else:
            logging.info("  %s = %s", key, str(eval_metrics[key]))

    # 3.保存模型
    model_path = pkg_resources.resource_filename(__name__, '../model/eda/')
    save_fast_model(fast_model, model_path)

    logging.info('结束')


