# !python3
# coding=utf-8
"""
-------------------------------------------------
   File Name：hetong_spark
   Description :
   Author : 小巫女
   date：2019/6/21
-------------------------------------------------
   Change Activity:
   2019/6/21 16:52
-------------------------------------------------
"""
import codecs
import os
import sys

import jieba
from pyspark.ml import Pipeline, PipelineModel
# from pyspark.mllib.linalg import Vectors
# from pyspark.mllib.feature import HashingTF, IDF
from pyspark.sql import SparkSession

from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, CountVectorizer, StringIndexer, IndexToString

# Path for spark source folder
os.environ['JAVA_HOME'] = 'C:\\Program Files\\Java\\jdk1.8.0_181'
os.environ['PATHONPATH'] = 'D:\\bigdata\\spark\\python'
os.environ['SPARK_HOME'] = "D:\\bigdata\\spark"

# Append pyspark to Python Path
sys.path.append("D:\\bigdata\\spark\\python")
sys.path.append("D:\\bigdata\\spark\\python\\lib\\py4j-0.10.4-src.zip")


# 分割关键词特征和标签类别，标签转换为id
def parse_line_DF(path):
    raw_data = sc.textFile(path)# read data
    numColumns = raw_data.count()# number of columns in dataset
    records = raw_data.map(lambda line: line.split(' '))
    data = records.collect()
    data1 = []
    pre_label = ''
    label_num = -1
    label_dict = {}
    for i in range(numColumns):#遍历dataset每一行
        trimmed = [each for each in data[i]]
        label = trimmed[-1]#类别标签
        if label != pre_label:
            label_num += 1
            print(str(label_num) + ':' + label)
            label_dict[label] = label_num
        features = list(map(lambda x: x, trimmed[0:len(trimmed) - 1]))#关键词特征
        # features = [StructField('f' + str(l), StringType(), True) for l in range(fieldnum)]
        # schema = StructType(features)
        c = (label_num, features)
        data1.append(c)
        pre_label = label
    training = myspark.createDataFrame(data1, ['label','words'])
    # training.cache()#缓存

    return training, label_dict


def parse_line_DF_nolabeltrans(path):
    raw_data = sc.textFile(path)# read data
    numColumns = raw_data.count()# number of columns in dataset
    records = raw_data.map(lambda line: line.split(' '))
    data = records.collect()
    data1 = []
    pre_label = ''
    label_num = -1
    label_dict = {}
    for i in range(numColumns):#遍历dataset每一行
        trimmed = [each for each in data[i]]
        label = trimmed[-1]#类别标签
        if label != pre_label:
            label_num += 1
            print(str(label_num) + ':' + label)
            label_dict[label] = label_num
        features = list(map(lambda x: x, trimmed[0:len(trimmed) - 1]))#关键词特征
        # features = [StructField('f' + str(l), StringType(), True) for l in range(fieldnum)]
        # schema = StructType(features)
        c = (label, features)
        data1.append(c)
        pre_label = label
    training = myspark.createDataFrame(data1, ['label','words'])
    # training.cache()#缓存

    return training, label_dict


def readIdf(words):

    # 读入TXT文件 #
    pathname = "D:\\合同\\合同分类\\数据\\train\\v1.1\\idf_hetong17.txt"
    file = codecs.open(pathname,'r',encoding='utf-8')
    # important_words = [w.strip() for w in codecs.open('word_weight.txt', 'r', encoding='utf-8').readlines()]
    tf_list = {}
    idf_list = {}
    tfIdf_list = {}
    lines = file.readlines()
    ## 计算每个词的tf并去重 ##
    for word in words:
        if word in tf_list.keys():
            tf_list[word] = tf_list[word]+1
        else:
            tf_list[word] = 1

    # 获取"idf.txt"中每个词的idf
    for line in lines:
        word1 = str(line).split(':')[0]
        idf = float(str(line).split(':')[1])
        idf_list[word1] = idf

    ## 计算测试文本中每个词的tf-idf ##
    for word in words:
        ratio = 1
        tf = tf_list[word]
        if word in idf_list.keys():
            idf = idf_list[word]
        else:
            idf = 0
        tfIdf = tf*idf  # tfidf值
        # 根据词重要性（法律人员给出的各罪名下的重要关键词），改进权重值
        # if word in important_words:
        #     ratio = 1.5
        tfIdf_ratio = tfIdf*ratio
        tfIdf_list[word] = tfIdf_ratio

    # 按值排序字典并取前50个关键词
    tfIdf_sorted_dict = dict(sorted(tfIdf_list.items(), key=lambda d: d[1], reverse=True))
    top_n = []
    # 取出前50个， 也可以在sorted返回的list中取前几个
    cnt = 0
    for key, value in tfIdf_sorted_dict.items():
        cnt += 1
        if cnt > 50:
            break
        # print("{}:{}".format(key, value))
        top_n.append(key)

    return top_n


# 对测试文本进行分割标签与正文，标签转换为对应的id，分词,jieba
def tokenize_DF(path, c2l):
    raw_data = sc.textFile(path)  # read data
    raw_data.cache()
    numColumns = raw_data.count()  # number of columns in dataset
    records = raw_data.map(lambda line: line.split('^^'))
    records.cache()
    data = records.collect()
    data1 = []
    pre_label = ''
    label_num = -1
    stopwords = [w.strip() for w in codecs.open('chinese_stopword.txt', 'r', encoding='utf-8').readlines()]
    for i in range(numColumns):#遍历dataset每一行
        trimmed = [each for each in data[i]]
        label = trimmed[-1]#类别标签
        #convert category name to id
        try:
            label_num = c2l.get(label)
            if label_num is None:
                label_num = 1
        except KeyError as e:
            label_num = 100
        # print(label_num)

        content = trimmed[0]
        words = jieba.lcut(content)#分词
        features = remove_stopwords(words, stopwords)#去停用词
        #tfidf提取前50个关键词
        new_features = readIdf(features)
        c = (label_num, new_features)
        data1.append(c)
        pre_label = label
    testing = myspark.createDataFrame(data1, ['label','words'])

    return testing


# 对测试文本进行分割标签与正文，标签不转换，分词,jieba
def tokenize_DF_nolabeltrans(path):
    raw_data = sc.textFile(path)  # read data
    raw_data.cache()
    numColumns = raw_data.count()  # number of columns in dataset
    records = raw_data.map(lambda line: line.split('^^'))
    records.cache()
    data = records.collect()
    data1 = []
    pre_label = ''
    stopwords = [w.strip() for w in codecs.open('chinese_stopword.txt', 'r', encoding='utf-8').readlines()]
    for i in range(numColumns):#遍历dataset每一行
        trimmed = [each for each in data[i]]
        label = trimmed[-1]#类别标签

        content = trimmed[0]
        words = jieba.lcut(content)#分词
        features = remove_stopwords(words, stopwords)#去停用词
        #tfidf提取前50个关键词
        new_features = readIdf(features)
        c = (label, new_features)
        data1.append(c)
        pre_label = label
    testing = myspark.createDataFrame(data1, ['label','words'])

    return testing


def remove_stopwords(wordlist, stopkey):
    # stopkey = [w.strip() for w in codecs.open('chinese_stopword.txt', 'r', encoding='utf-8').readlines()]
    newlist = []
    for word in wordlist:
        if word not in stopkey and (word.find(' ') is -1) and (word is not '') and (word.find('　') is -1):
            newlist.append(word)
    return newlist


def train_nb_model_tfidf():
    training_df, label_dict = parse_line_DF('Train17_keyword.txt')
    # testing_df = parse_line_DF('Train17_keyword.txt')
    training_df.show(10)
    training_df.cache()
    # tokenizer = Tokenizer().setInputCol("text").setOutputCol("words")
    hashingTF = HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(21000)
    idf = IDF().setInputCol("rawFeatures").setOutputCol("features")
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    pipeline = Pipeline().setStages([hashingTF, idf, nb])
    model = pipeline.fit(training_df)
    model.write().overwrite().save('hetong-NB-model-tfidf')
    return label_dict


def train_nb_model_frequency():
    training_df, label_dict = parse_line_DF('Train17_keyword.txt')
    # testing_df = parse_line_DF('Train17_keyword.txt')
    training_df.show(10)
    training_df.cache()
    # tokenizer = Tokenizer().setInputCol("text").setOutputCol("words")
    # hashingTF = HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(100)
    # idf = IDF().setInputCol("rawFeatures").setOutputCol("features")
    count_vectors = CountVectorizer(inputCol='words', outputCol='features', vocabSize=21000, minDF=2)
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    pipeline = Pipeline().setStages([count_vectors, nb])
    model = pipeline.fit(training_df)
    model.write().overwrite().save('hetong-NB-model-frequency2')
    return label_dict


def train_nb_model_frequency_labelindex():
    training_df, label_dict = parse_line_DF_nolabeltrans('Train17_keyword.txt')
    # testing_df = parse_line_DF('Train17_keyword.txt')
    training_df.show(10)
    training_df.cache()
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel",handleInvalid="keep")
    # tokenizer = Tokenizer().setInputCol("text").setOutputCol("words")
    # hashingTF = HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(100)
    # idf = IDF().setInputCol("rawFeatures").setOutputCol("features")
    count_vectors = CountVectorizer(inputCol='words', outputCol='features', vocabSize=21000, minDF=2)
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial", labelCol='indexedLabel')
    pipeline = Pipeline().setStages([labelIndexer, count_vectors, nb])
    model = pipeline.fit(training_df)
    model.write().overwrite().save('hetong-NB-model-frequency2')
    return label_dict


def train_rf_model_frequency_labelindex():
    training_df, label_dict = parse_line_DF_nolabeltrans('Train17_keyword.txt')
    # testing_df = parse_line_DF('Train17_keyword.txt')
    training_df.show(10)
    training_df.cache()
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel",handleInvalid="keep")
    # tokenizer = Tokenizer().setInputCol("text").setOutputCol("words")
    # hashingTF = HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(100)
    # idf = IDF().setInputCol("rawFeatures").setOutputCol("features")
    count_vectors = CountVectorizer(inputCol='words', outputCol='features', vocabSize=21000, minDF=2)
    rf = RandomForestClassifier(numTrees=150, featureSubsetStrategy="auto", impurity='gini', maxDepth=30, labelCol='indexedLabel')
    pipeline = Pipeline().setStages([labelIndexer, count_vectors, rf])
    model = pipeline.fit(training_df)
    model.write().overwrite().save('hetong-RF-model-frequency2')
    return label_dict


def train_rf_model_frequency():
    training_df, label_dict = parse_line_DF('Train17_keyword.txt')
    # testing_df = parse_line_DF('Train17_keyword.txt')
    training_df.show(10)
    # training_df.cache()
    # tokenizer = Tokenizer().setInputCol("text").setOutputCol("words")
    # hashingTF = HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(100)
    # idf = IDF().setInputCol("rawFeatures").setOutputCol("features")
    count_vectors = CountVectorizer(inputCol='words', outputCol='features', vocabSize=21000, minDF=2)
    rf = RandomForestClassifier(numTrees=150, featureSubsetStrategy="auto", impurity='gini', maxDepth=30)
    pipeline = Pipeline().setStages([count_vectors, rf])
    model = pipeline.fit(training_df)
    model.write().overwrite().save('hetong-RF-model-frequency')
    return label_dict


def train_rf_model_tfidf():
    training_df, label_dict = parse_line_DF('Train17_keyword.txt')
    # testing_df = parse_line_DF('Train17_keyword.txt')
    training_df.show(10)
    # training_df.cache()
    # tokenizer = Tokenizer().setInputCol("text").setOutputCol("words")
    hashingTF = HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(21000)
    idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(2)
    # count_vectors = CountVectorizer(inputCol='words', outputCol='features', vocabSize=21000, minDF=2)
    rf = RandomForestClassifier(numTrees=10, featureSubsetStrategy="auto", impurity='gini', maxDepth=15)
    pipeline = Pipeline().setStages([hashingTF, idf, rf])
    model = pipeline.fit(training_df)
    model.write().overwrite().save('hetong-RF-model-tfidf')
    return label_dict


def test_nb_model(c2l):
    sameModel = PipelineModel.load('hetong-RF-model-frequency') #hetong-NB-model hetong-NB-model-frequency hetong-RF-model-frequency
    # 对测试文本进行分词,jieba
    testing_df = tokenize_DF('test_all.txt',c2l)
    testing_df.cache()
    testing_df.show(10)
    print(testing_df.collect()[-1])
    prediction = sameModel.transform(testing_df)
    print(prediction.collect()[-1])
    result_comparison = prediction.select('label','prediction').collect()#单独拎出来标签和预测结果
    result_prob = prediction.select('rawPrediction','probability').collect()#各类别预测概率
    write2file = codecs.open('result-rf-freq.txt','w',encoding='utf-8')  # result-freq
    write2file2 = codecs.open('result-rf-freq-prob.txt', 'w', encoding='utf-8')
    for result in result_comparison:
        write2file.write(str(result[0]) + '\t' + str(result[1]) + '\n')
    for prob in result_prob:
        write2file2.write(str(prob[0]) + '\t\t' + str(prob[1]) + '\n')
    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction')
    print(evaluator.evaluate(prediction))
    write2file.close()


def test_nb_model_labelindex():
    sameModel = PipelineModel.load('hetong-NB-model-frequency2') #hetong-NB-model hetong-NB-model-frequency hetong-RF-model-frequency
    # # 打印特征重要性
    # print(sameModel.stages[2].featureImportances)
    # 对测试文本进行分词,jieba
    testing_df = tokenize_DF_nolabeltrans('test_all2.txt')
    testing_df.cache()
    testing_df.show(10)
    print(testing_df.collect()[-1])
    # 模型预测
    prediction = sameModel.transform(testing_df)
    print(prediction.collect()[-1])
    result_comparison = prediction.select('indexedLabel','prediction').collect()#单独拎出来标签索引和预测结果
    result_prob = prediction.select('rawPrediction','probability').collect()#各类别预测概率

    write2file = codecs.open('result-nb-freq2.txt','w',encoding='utf-8')  # result-freq
    write2file2 = codecs.open('result-nb-freq-prob2.txt', 'w', encoding='utf-8')
    # 转换预测类别索引为类别名称
    index_converter = IndexToString(inputCol="prediction", outputCol="predictLabel", labels=['001', '025', '012', '003', '023', '010', '007', '026', '006', '009', '013', '021', '017', '005', '022', '027', '008'])
    label_converted = index_converter.transform(prediction)
    print(label_converted.show(5))
    # 写入结果到txt文件
    for result in result_comparison:
        write2file.write(str(result[0]) + '\t' + str(result[1]) + '\n')
    for prob in result_prob:
        write2file2.write(str(prob[0]) + '\t\t' + str(prob[1]) + '\n')

    # 评估器计算准确率
    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='indexedLabel')
    print(evaluator.evaluate(prediction))
    write2file.close()
    write2file2.close()


def test_rf_model_labelindex():
    sameModel = PipelineModel.load('hetong-RF-model-frequency2') #hetong-NB-model hetong-NB-model-frequency hetong-RF-model-frequency
    # 打印特征重要性
    print(sameModel.stages[2].featureImportances)
    # 取出中间步骤stringIndexer的参数, 所有label名称
    labelIndexer = sameModel.stages[0].labels
    print(labelIndexer)
    # 对测试文本进行分词,jieba
    testing_df = tokenize_DF_nolabeltrans('test_all2.txt')
    testing_df.cache()
    testing_df.show(10)
    print(testing_df.collect()[-1])
    # 模型预测
    prediction = sameModel.transform(testing_df)
    print(prediction.collect()[-1])
    result_comparison = prediction.select('indexedLabel','prediction').collect()#单独拎出来标签索引和预测结果
    result_prob = prediction.select('rawPrediction','probability').collect()#各类别预测概率

    write2file = codecs.open('result-rf-freq2.txt','w',encoding='utf-8')  # result-freq
    write2file2 = codecs.open('result-rf-freq-prob2.txt', 'w', encoding='utf-8')
    # 转换预测类别索引为类别名称
    index_converter = IndexToString(inputCol="prediction", outputCol="predictLabel", labels=labelIndexer)
    label_converted = index_converter.transform(prediction)
    print(label_converted.show(5))
    # 写入结果到txt文件
    for result in result_comparison:
        write2file.write(str(result[0]) + '\t' + str(result[1]) + '\n')
    for prob in result_prob:
        write2file2.write(str(prob[0]) + '\t\t' + str(prob[1]) + '\n')

    # 评估器计算准确率
    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='indexedLabel')
    print(evaluator.evaluate(prediction))
    write2file.close()
    write2file2.close()


def accuracy_cal():
    read_result = codecs.open('result-rf-freq.txt','r',encoding='utf-8')
    write_accu = codecs.open('accuracy-rf-freq.txt','w',encoding='utf-8')
    lines = read_result.readlines()
    prev_truth = -1
    count = 0
    right_list = {}
    sum_result = {}
    for line in lines:
        truth = int(line.split('\t')[0])
        prediction = int(line.split('\t')[1].replace('.0\n',''))
        if truth == prev_truth:
            count += 1
        else:
            count = 1
        if truth == prediction:
            accu = 1
            if right_list.get(truth) is None:
                right_list[truth] = 1
            else:
                right_list[truth] = right_list[truth] + 1
        else:
            accu = 0

        # write_accu.write(truth + " " + prediction + " " + accu + "\n");
        prev_truth = truth
        sum_result[prev_truth] = count

    right_list1 = sorted(right_list.items(),key=lambda x:x[1])
    for tup in right_list1:
        write_accu.write(str(tup[0]) + ' = ' + str(tup[1]/sum_result[tup[0]]) + '\n')
    write_accu.close()


if __name__ == '__main__':
    myspark = SparkSession.builder \
        .appName('Spark_SQL_basic_example') \
        .config('spark.executor.memory', '8g') \
        .config('spark.driver.memory', '8g') \
        .getOrCreate()
    sc = myspark.sparkContext
    cate2label = {}

    # idfModel = pipeline.fit(training_df)
    # rescaledData = idfModel.transform(training_df)
    # printData = rescaledData.select("label", "words", "features")
    # print(printData.take(3))
    # # trainDataRdd = rescaledData.select("category","features").map()
    # trainDataRdd = rescaledData.select("label", "features")  # .rdd.map(lambda row: LabeledPoint(row[0], row[1:]))
    # print(trainDataRdd.count())
    # print(trainDataRdd.printSchema())
    # print(trainDataRdd.collect()[-1])
    # model = NaiveBayes().modelType('').fit(trainDataRdd)

    # sameModel = NaiveBayesModel.load('hetong-NB-model')

    # testrescaledData = idfModel.transform(srcDF2)
    # cate2label = train_nb_model_tfidf()
    # cate2label = train_nb_model_frequency_labelindex()
    # cate2label = train_rf_model_frequency()
    # cate2label = train_rf_model_frequency_labelindex()
    # cate2label = train_rf_model_tfidf()

    # lines = [w.strip() for w in codecs.open('id2category.txt', 'r', encoding='utf-8').readlines()]
    # for line in lines:
    #     labid = int(line.split(':')[0])
    #     catename = line.split(':')[1]
    #     cate2label[catename] = labid
    # test_nb_model(cate2label)
    # test_nb_model_labelindex()
    test_rf_model_labelindex()
    # accuracy_cal()



