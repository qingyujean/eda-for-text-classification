# eda-for-text-classification
使用EDA技术对小型的不均衡的数据集做增强，验证其效果提升

## 说明
* 数据集有3列，使用pandas读取，`label`列是文本类别标记，`clear_content`是在最原始爬取的数据上做了清洗了，去除了非常用的标点以及其他特殊符号，`content_seg`是在`clear_content`上进一步做了分词。此次实验主要使用了`label`和`content_seg`列。
* 实验中会使用`fasttext`在`原始数据集`和`增强后的数据集`上分别训练，最后给出结果对比。
* EDA代码使用了这里的实现：[https://github.com/zhanlaoban/eda_nlp_for_Chinese](https://github.com/zhanlaoban/eda_nlp_for_Chinese)
* 从结果中的`混下矩阵`来看，进行增强后，分类性能有显著提升，`f1`值也从原先的`0.21`提升到`0.80`，在增强后的数据上看，模型有些许的过拟合，但总体生来看，模型比无数据增强时，已经有了质的飞跃。
* 关于fasttext的模型参数的选择，我使用的是`网格搜索+交叉验证`

## 参考
[1] [eda 论文](https://arxiv.org/abs/1901.11196)

[2] [eda在中文语料的实现（github）](https://github.com/zhanlaoban/eda_nlp_for_Chinese)
