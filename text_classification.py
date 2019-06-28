# 中文文本分类
import os
import jieba
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

warnings.filterwarnings('ignore')

def cut_words(file_path):
    """
    对文本进行切词
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text_with_spaces = ''
    text=open(file_path, 'r', encoding='gb18030').read()
    textcut = jieba.cut(text)
    for word in textcut:
        text_with_spaces += word + ' '
    return text_with_spaces

def loadfile(file_dir, label):
    """
    将路径下的所有文件加载
    :param file_dir: 保存txt文件目录
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    """
    file_list = os.listdir(file_dir)
    words_list = []
    labels_list = []
    for file in file_list:
        file_path = file_dir + '/' + file
        words_list.append(cut_words(file_path))
        labels_list.append(label)                                                                                                                 
    return words_list, labels_list

# 训练数据
train_words_list1, train_labels1 = loadfile('text classification/train/女性', '女性')
train_words_list2, train_labels2 = loadfile('text classification/train/体育', '体育')
train_words_list3, train_labels3 = loadfile('text classification/train/文学', '文学')
train_words_list4, train_labels4 = loadfile('text classification/train/校园', '校园')

train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4

# 测试数据
test_words_list1, test_labels1 = loadfile('text classification/test/女性', '女性')
test_words_list2, test_labels2 = loadfile('text classification/test/体育', '体育')
test_words_list3, test_labels3 = loadfile('text classification/test/文学', '文学')
test_words_list4, test_labels4 = loadfile('text classification/test/校园', '校园')

test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4

stop_words = open('text classification/stop/stopword.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff处理
stop_words = stop_words.split('\n') # 根据分隔符分隔

# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

train_features = tf.fit_transform(train_words_list)
# 上面fit过了，这里transform
test_features = tf.transform(test_words_list) 

# 多项式贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB  
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
predicted_labels=clf.predict(test_features)

# 计算准确率
print('准确率为：', metrics.accuracy_score(test_labels, predicted_labels))