# eda-for-text-classification
使用EDA技术对小型的不均衡的数据集做增强，验证其效果提升

## 更新
2020-12-30-----
由于很多人反应 **words.vector.gz** 这个下载不到了，找我email要这个文件，所以特意将此文件上传到github，方便大家下载。存放路径在：./EDA/files/下的data.rar文件

## 说明
* 数据集有3列，使用pandas读取，`label`列是文本类别标记，`clear_content`是在最原始爬取的数据上做了清洗了，去除了非常用的标点以及其他特殊符号，`content_seg`是在`clear_content`上进一步做了分词。此次实验主要使用了`label`和`content_seg`列。
* 实验中会使用`fasttext`在`原始数据集`和`增强后的数据集`上分别训练，最后给出结果对比。
* EDA代码使用了这里的实现：[https://github.com/zhanlaoban/eda_nlp_for_Chinese](https://github.com/zhanlaoban/eda_nlp_for_Chinese)
* 从结果中的`混下矩阵`来看，进行增强后，分类性能有显著提升，`f1`值也从原先的`0.21`提升到`0.80`，在增强后的数据上看，模型有些许的过拟合，但总体生来看，模型比无数据增强时，已经有了质的飞跃。
* 关于fasttext的模型参数的选择，我使用的是`网格搜索+交叉验证`

## 代码结构
* fast/training_data_analysis.py 对训练数据集的样本分布做了简要的分析
* fast/fasttext_train.py 和 fast/fasttext_test.py 在无增强的原数据集上做训练和测试，原始数据集存放于data/下
* augment/gene_eda_data.py 主要对训练数据集和测试数据集应用EDA技术做数据增强，增强后的数据存放于data/augment/下；核心eda实现调用的是augment/eda.py
* fast/fasttext_tune.py 主要是使用“网格搜索”和“交叉验证”技术，对fasttext进行调参/参数选择
* fast/fasttext_train_with_eda.py 和 fast/fasttext_test_with_eda.py 在做了eda增强后的新数据集上做训练和测试
* 其他文件夹：
  * data/fast/以及data/fast/augment/存放处理成fasttext训练数据格式的数据
  * data/zh_data/存放停用词表，eda中会使用到
  * model/存放训练的模型

## 参考
[1] [eda 论文](https://arxiv.org/abs/1901.11196)

[2] [eda在中文语料的实现（github）](https://github.com/zhanlaoban/eda_nlp_for_Chinese)
