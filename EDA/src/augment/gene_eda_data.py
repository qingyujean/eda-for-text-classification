# coding=utf-8
"""
Author  : Jane
Contact : xijian@ict.ac.cn
Time    : 2020/9/27 15:43
Desc:
"""

import pandas as pd
from sklearn.utils import shuffle
from augment.eda import eda


DATA_PATH = '../data/'




def get_eda_df(sentences, alpha=0.1, num_avg=9):
    results = []
    for i, sents in enumerate(sentences):
        augmented_sentences = eda(sents, segged=True, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha,
                                  num_aug=num_avg)
        results.append(augmented_sentences)
    return sum(results, [])




def data_augmented(filepath):
    df = pd.read_csv(filepath, encoding='UTF-8', sep='\t', header=0, index_col=False, usecols=[0, 2])
    print(df.head())
    print(df.shape)  # (343, 2)
    print(df.label.value_counts())
    df.label.value_counts().plot(kind='barh')

    # 对各类别数据分别做增强，增强的次数与原始样本数量有关，原始样本少，就相应生成更多的增强样本。
    df_zgbd = df[df['label'] == 2]  # 
    print(df_zgbd.shape)  # (11, 2)
    df_zgbd_augmented = pd.DataFrame({'content_seg_augmented': get_eda_df(df_zgbd['content_seg'], alpha=0.05, num_avg=20)})
    # print(len(df_zgbd_augmented), df_zgbd_augmented[:5])
    df_zgbd_augmented['label'] = [2] * len(df_zgbd_augmented)
    print(df_zgbd_augmented.shape)  # (11*(20+1)=231, 2)
    
    df_jszc = df[df['label'] == 3]  # 
    print(df_jszc.shape)  # (11, 2)
    df_jszc_augmented = pd.DataFrame({'content_seg_augmented': get_eda_df(df_jszc['content_seg'], alpha=0.05, num_avg=20)})
    df_jszc_augmented['label'] = [3] * len(df_jszc_augmented)
    print(df_jszc_augmented.shape)  # (11*(20+1)=231, 2)
    
    df_jsyx = df[df['label'] == 1]  # 
    print(df_jsyx.shape)  # (43, 2)
    df_jsyx_augmented = pd.DataFrame({'content_seg_augmented': get_eda_df(df_jsyx['content_seg'], alpha=0.05, num_avg=5)})
    df_jsyx_augmented['label'] = [1] * len(df_jsyx_augmented)
    print(df_jsyx_augmented.shape)  # (43*(5+1)=258, 2)
    
    df_bushu = df[df['label'] == 4]  # 
    print(df_bushu.shape)  # (55, 2)
    df_bushu_augmented = pd.DataFrame({'content_seg_augmented': get_eda_df(df_bushu['content_seg'], alpha=0.05, num_avg=3)})
    df_bushu_augmented['label'] = [4] * len(df_bushu_augmented)
    print(df_bushu_augmented.shape)  # (55*(3+1)=220, 2)
    
    df_wqyf = df[df['label'] == 6]  # 
    print(df_wqyf.shape)  # (94, 2)
    df_wqyf_augmented = pd.DataFrame({'content_seg_augmented': get_eda_df(df_wqyf['content_seg'], alpha=0.05, num_avg=2)})
    df_wqyf_augmented['label'] = [6] * len(df_wqyf_augmented)
    print(df_wqyf_augmented.shape)  # (94*(2+1)=282, 2)
    
    df_qyct = df[df['label'] == 5]  # 
    print(df_qyct.shape)  # (129, 2)
    df_qyct_augmented = pd.DataFrame({'content_seg_augmented': get_eda_df(df_qyct['content_seg'], alpha=0.05, num_avg=1)})
    df_qyct_augmented['label'] = [5] * len(df_qyct_augmented)
    print(df_qyct_augmented.shape)  # (129*(1+1)=258, 2)
    
    df_augmented = pd.concat([df_zgbd_augmented, df_jszc_augmented, df_jsyx_augmented,
                              df_bushu_augmented, df_wqyf_augmented, df_qyct_augmented], axis=0)  # 还没加上df_0_and_7，
    
    df_augmented = shuffle(df_augmented)
    
    print(df_augmented.shape)  # (1480, 2)
    print(df_augmented.label.value_counts())
    df_augmented.label.value_counts().plot(kind='barh')
    
    return df_augmented
    
    
# 处理训练数据，数据格式：label    clear_content    content_seg
filepath = DATA_PATH + 'js_pd_tagged_train.txt'
df_augmented = data_augmented(filepath)
df_augmented.to_csv(DATA_PATH + 'augmented/js_pd_tagged_train.txt', sep='\t', index=False, encoding='UTF-8')


filepath = DATA_PATH + 'js_pd_tagged_test.txt'
df_augmented = data_augmented(filepath)
df_augmented.to_csv(DATA_PATH + 'augmented/js_pd_tagged_test.txt', sep='\t', index=False, encoding='UTF-8')
