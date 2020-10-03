#coding=utf-8

'''
Created on 2020-4-20

@author: Yoga
'''

import pandas as pd
from matplotlib import pyplot as plt


DATA_PATH = '../data/'


if __name__=='__main__':
    #数据格式：label    clear_content    content_seg
    #训练集
    df = pd.read_csv(DATA_PATH+'js_pd_tagged_train.txt', encoding='UTF-8', sep = '\t', header=0, index_col=False)
    print(df.head())
    print(df.shape)
    print(df['label'].value_counts())
    df.label.value_counts().plot(kind='barh')
    plt.show()
