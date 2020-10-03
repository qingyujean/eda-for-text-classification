#coding=utf-8

'''
Created on 2020-4-20

@author: Yoga
'''
import pkg_resources
import logging
import pandas as pd
import numpy as np

from fast import get_metrics, create_faststyle_inputdata, load_fast_model, fast_predict

DATA_PATH = '../data/'
logging.basicConfig(level=logging.INFO)


if __name__=='__main__':
    # 处理测试数据，变成fasttext认识的格式
    # 增强后的数据 content_seg_augment	label
    df_test = create_faststyle_inputdata(DATA_PATH + 'augmented/js_pd_tagged_test.txt', [0, 1])
    df_test.to_csv(DATA_PATH+'fast/augmented/js_pd_tagged_test.txt', sep='\t', index=False, header=None, encoding='UTF-8')  # 不要表头

    # 加载fasttext模型
    model_name = 'fast_20201003_143955.pkl' # 增强后的模型
    model_path = pkg_resources.resource_filename(__name__, "../model/eda/"+model_name)
    fast_model = load_fast_model(model_path)

    # 测试
    # 加载测试数据
    df_test_fast = pd.read_csv(DATA_PATH+'fast/augmented/js_pd_tagged_test.txt', header=None, encoding='UTF-8', sep='\t', index_col=False)  #没有表头
    X_test = [str(x).strip() for x in df_test_fast[0]] 
    y_test = [int(cls[-1:]) for cls in df_test_fast[1]]
    y_pred, y_pred_prob = fast_predict(fast_model, X_test, k=1)  #参数 k=1为默认值

    # 计算评价指标
    n_classes = len(np.unique(y_test))
    eval_metrics = get_metrics(y_test, y_pred, n_classes)
    logging.info('*'*15 + ' eval results ' + '*'*15)
    for key in sorted(eval_metrics.keys()):
        if key in ('confusion_matrix', 'classification_report'):
            logging.info("  %s = \n%s", key, str(eval_metrics[key]))
        else:
            logging.info("  %s = %s", key, str(eval_metrics[key]))
    
    logging.info('结束')
