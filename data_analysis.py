# !python3
# coding=utf-8
"""
-------------------------------------------------
   File Name：data_analysis
   Description :
   Author : 小巫女
   date：2019/6/14
-------------------------------------------------
   Change Activity:
   2019/6/14 13:21
-------------------------------------------------
"""
import os
import sys

import pysnooper
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
# from pyspark.mllib.linalg import Vectors
# from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.stat import Statistics
from pyspark.sql import SparkSession

from pyspark.context import SparkContext
from pyspark.ml.classification import DecisionTreeClassifier
# import pyspark.ml.classification as cl
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer, IDF, VectorIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, StructField, StructType

# Path for spark source folder
os.environ['JAVA_HOME'] = 'C:\\Program Files\\Java\\jdk1.8.0_181'
os.environ['PATHONPATH'] = 'D:\\bigdata\\spark\\python'
os.environ['SPARK_HOME'] = "D:\\bigdata\\spark"

# Append pyspark to Python Path
sys.path.append("D:\\bigdata\\spark\\python")
sys.path.append("D:\\bigdata\\spark\\python\\lib\\py4j-0.10.4-src.zip")


@pysnooper.snoop()
def test1():
    sc = SparkContext('local', 'test')
    rdd = sc.textFile('data')
    tf = HashingTF()
    tfVectors = tf.transform(rdd).cache()
    idf = IDF()
    idfModel = idf.fit(tfVectors)
    tfIdfVectors = idfModel.transform(tfVectors)
    # print(tfVectors.collect())
    # print(tfIdfVectors.collect())
    Statistics.colStats(tfIdfVectors)


def ml_package():
    #### ml包基本工具的实例 ####
    myspark = SparkSession.builder \
        .appName('Spark_SQL_basic_example') \
        .config('spark.executor.memory', '2g') \
        .getOrCreate()
    dataTr = [(1.0, Vectors.dense(0.0, 1.1, 0.1)),
              (0.0, Vectors.dense(2.0, 1.0, -1.0)),
              (0.0, Vectors.dense(2.0, 1.3, 1.0)),
              (1.0, Vectors.dense(0.0, 1.2, -0.5))]
    trainingData = myspark.createDataFrame(dataTr).toDF('label', 'features')
    trainingData.show()
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    # print('LogisticRegression parameters:\n' + lr.explainParams() + '\n')
    model1 = lr.fit(trainingData)
    print('model 1 was fit using params: ')
    print(model1.extractParamMap())
    # 修改其中的一个参数
    # We may alternatively specify parameters using a Python dictionary as a paramMap
    paramMap = {lr.maxIter: 20}
    # 覆盖掉
    paramMap[lr.maxIter] = 30  # Specify 1 Param, overwriting the original maxIter.
    # 更新参数对
    paramMap.update({lr.regParam: 0.1, lr.threshold: 0.55})  # Specify multiple Params.
    # You can combine paramMaps, which are python dictionaries.
    # 新的参数，合并为两组参数对
    paramMap2 = {lr.probabilityCol: "myProbability"}  # Change output column name
    paramMapCombined = paramMap.copy()
    paramMapCombined.update(paramMap2)
    model2 = lr.fit(trainingData, paramMapCombined)
    print("Model 2 was fit using parameters: ")
    print(model2.extractParamMap())
    dataTs = [(1.0, Vectors.dense(-1.0, 1.5, 1.3)),
              (0.0, Vectors.dense(3.0, 2.0, -0.1)),
              (1.0, Vectors.dense(0.0, 2.2, -1.5))]
    testData = myspark.createDataFrame(dataTs).toDF('label', 'features')
    testData.show()
    prediction = model2.transform(testData)
    selected = prediction.select("features", "label", "myProbability", "prediction")
    for row in selected.collect():
        print(row)


def pipeline_example():
    #### pipeline的实例 ####
    myspark = SparkSession.builder \
        .appName('Spark_SQL_basic_example') \
        .config('spark.executor.memory', '2g') \
        .getOrCreate()
    dataTr = [(0, "a b c d e spark", 1.0),
              (1, "你 她", 0.0),
              (2, "spark f g 我", 1.0),
              (3, "hadoop mapreduce", 0.0)]
    trainingData = myspark.createDataFrame(dataTr).toDF('id', 'text', 'label')
    trainingData.show()
    tokenizer = Tokenizer().setInputCol('text').setOutputCol('words')
    hashingTF = HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol()).setOutputCol('features')

    lr = LogisticRegression().setMaxIter(10).setRegParam(0.001)
    pipeline = Pipeline().setStages([tokenizer, hashingTF, lr])
    model = pipeline.fit(trainingData)
    model.write().overwrite().save('spark-logistic-regression-model')
    pipeline.write().overwrite().save('unfit-lr-model')

    sameModel = PipelineModel.load('spark-logistic-regression-model')

    # 测试数据
    # Prepare test documents, which are unlabeled (id, text) tuples.
    test = myspark.createDataFrame([
        (4, "spark i j k"),
        (5, "l m n"),
        (6, "mapreduce spark"),
        (7, "apache hadoop"),
        (8, "o p q r 我")], ["id", "text"])
    # 预测，打印出想要的结果
    # Make predictions on test documents and print columns of interest.
    prediction = sameModel.transform(test)
    selected = prediction.select("id", "text", "prediction")
    for row in selected.collect():
        print(row)


def decision_tree_example1():
    myspark = SparkSession.builder \
        .appName('Spark_SQL_basic_example') \
        .config('spark.executor.memory', '4g') \
        .getOrCreate()
    rawData = myspark.sparkContext.textFile('test.csv')
    lines = rawData.map(lambda x: x.split(','))
    fieldnum = len(lines.first())
    print(lines.count())
    print(fieldnum)
    fields = [StructField('f'+str(l), StringType(), True) for l in range(fieldnum)]
    schema = StructType(fields)
    covtype_df = myspark.createDataFrame(lines, schema)
    print(covtype_df)
    covtype_df.show()

    # convert to double
    covtype_df = covtype_df.select([col(column).cast('double').alias(column) for column in covtype_df.columns])
    covtype_df.show()
    featuresCols = covtype_df.columns[:2]
    print(featuresCols)
    covtype_df = covtype_df.withColumn('label',covtype_df['f2']).drop('f2')  # 最后一列设置为label字段
    covtype_df.show()


# @pysnooper.snoop()
def decision_tree_example2():
    myspark = SparkSession.builder \
        .appName('Spark_SQL_basic_example') \
        .config('spark.executor.memory', '4g') \
        .getOrCreate()
    raw_data = myspark.sparkContext.textFile('test.csv')
    print(raw_data.take(1))
    numColumns = raw_data.count()
    print(numColumns)
    records = raw_data.map(lambda line: line.split(','))
    data = records.collect()
    data1 = []
    for i in range(numColumns):
        trimmed = [each for each in data[i]]
        label = int(trimmed[-1])
        features = list(map(lambda x: x, trimmed[0:len(trimmed)-1]))
        c = (label, Vectors.dense(features))
        data1.append(c)
    df = myspark.createDataFrame(data1, ['label','features'])
    df.show()
    df.printSchema()
    df.cache()
    featureIndexer = VectorIndexer(inputCol='features',outputCol='indexedFeatures',maxCategories=24).fit(df)#建立特征索引
    (trainingData, testData) = df.randomSplit([0.9,0.1],seed=234)#切分训练集和测试集
    print(trainingData.count())
    print(testData.count())
    dt = DecisionTreeClassifier(maxDepth=5, labelCol='label', featuresCol='indexedFeatures', impurity='entropy')
    pipeline = Pipeline(stages=[featureIndexer, dt])
    model = pipeline.fit(trainingData)
    model.write().overwrite().save('spark_decision_tree_model')#存储模型
    samemodel = PipelineModel.load('spark_decision_tree_model')#加载模型
    predictedResultAll = samemodel.transform(testData)
    testData.show()
    predictedResultAll.select('prediction').show()

    df_prediction = predictedResultAll.select('prediction').toPandas()#将预测值转换为pandas
    dtPredictions = list(df_prediction.prediction)#转换为列表
    print(dtPredictions[:10])
    dtTotalCorrect = 0
    testRaw = testData.count()
    testLabel = testData.select('label').collect()
    for i in range(testRaw):
        if dtPredictions[i] == testLabel[i][0]:
            dtTotalCorrect += 1
    print(dtTotalCorrect)
    accuracy = 1.0 * dtTotalCorrect / testRaw  # 计算准确率
    print(accuracy)


def decision_tree_example_withcrossvali():
    myspark = SparkSession.builder \
        .appName('Spark_SQL_basic_example') \
        .config('spark.executor.memory', '4g') \
        .getOrCreate()
    raw_data = myspark.sparkContext.textFile('test.csv')
    print(raw_data.take(1))
    numColumns = raw_data.count()
    print(numColumns)
    records = raw_data.map(lambda line: line.split(','))
    data = records.collect()
    data1 = []
    for i in range(numColumns):
        trimmed = [each for each in data[i]]
        label = int(trimmed[-1])
        features = list(map(lambda x: x, trimmed[0:len(trimmed)-1]))
        c = (label, Vectors.dense(features))
        data1.append(c)
    df = myspark.createDataFrame(data1, ['label','features'])
    df.show()
    df.printSchema()
    df.cache()
    featureIndexer = VectorIndexer(inputCol='features',outputCol='indexedFeatures',maxCategories=24).fit(df)#建立特征索引
    (trainingData, testData) = df.randomSplit([0.9,0.1])#切分训练集和测试集
    print(trainingData.count())
    print(testData.count())

    dt = DecisionTreeClassifier(maxDepth=5, labelCol='label', featuresCol='indexedFeatures', impurity='entropy', maxBins=30)
    pipeline = Pipeline(stages=[featureIndexer, dt])

    # 初始化评估器
    evaluator = MulticlassClassificationEvaluator()
    # 设置参数网格
    paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [4,5,6]).build()
    # 设置交叉验证的参数
    cv = CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)

    cvmodel = cv.fit(trainingData)
    # cvmodel.write().overwrite().save('spark_decision_tree_model')#存储模型
    # samemodel = PipelineModel.load('spark_decision_tree_model')#加载模型
    predictedResultAll = cvmodel.transform(testData)
    testData.show()
    predictedResultAll.select('prediction').show()

    df_prediction = predictedResultAll.select('prediction').toPandas()#将预测值转换为pandas
    dtPredictions = list(df_prediction.prediction)#转换为列表
    print(dtPredictions[:10])
    dtTotalCorrect = 0
    testRaw = testData.count()
    testLabel = testData.select('label').collect()
    for i in range(testRaw):
        if dtPredictions[i] == testLabel[i][0]:
            dtTotalCorrect += 1
    print(dtTotalCorrect)
    accuracy = 1.0 * dtTotalCorrect / testRaw  # 计算准确率
    print(accuracy)
    bestmodel = cvmodel.bestModel.stages[1]  # 查看最匹配模型
    print(bestmodel.depth)


def tfidf():
    myspark = SparkSession.builder \
        .appName('Spark_SQL_basic_example') \
        .config('spark.executor.memory', '4g') \
        .getOrCreate()
    sentenceData = myspark.createDataFrame([(0,'Hi i heard about spark'),
                                            (0,'I wish Java could use case classes'),
                                            (1,'Logistic regression models are neat')]).toDF('label','sentence')
    tokenizer = Tokenizer().setInputCol('sentence').setOutputCol('words')
    wordsData = tokenizer.transform(sentenceData)#分词
    hashingTF = HashingTF().setInputCol('words').setOutputCol('rawFeatures').setNumFeatures(20)
    featurizedData = hashingTF.transform(wordsData)#tf,词频
    idf = IDF().setInputCol('rawFeatures').setOutputCol('features')
    idfModel = idf.fit(featurizedData)#idf
    rescaledData = idfModel.transform(featurizedData)
    rescaledData.show()





if __name__ == '__main__':
    # df = pd.read_csv('test.csv', header=0)
    # # print(df.count())
    # print(df.describe())
    # plt.figure()
    # bp = df.boxplot(return_type='dict')
    # x = bp['fliers'][0].get_xdata()
    # y = bp['fliers'][0].get_ydata()
    # y.sort()
    # # print(x, y)
    # plt.show()

    # denseVec1 = array([1.0, 2.0, 3.0])
    # denseVec2 = Vectors.dense([1.0, 2.0, 3.0])
    # sparseVec1 = Vectors.sparse(4, {0: 1.0, 2: 2.0})
    # sparseVec2 = Vectors.sparse(4, [0, 2], [1.0, 2.0])
    # print(denseVec1,denseVec2,sparseVec1,sparseVec2)

    # sentence = 'hello hello world'
    # words = sentence.split()
    # tf = HashingTF(10000)
    # sparseVec = tf.transform(words)
    # print(sparseVec)
    # sc = SparkContext('local', 'test')
    # rdd = sc.textFile('data')
    # tfVectors = tf.transform(rdd)
    # print(tfVectors.collect())

    # test1()

    # myspark = SparkSession.builder\
    #     .appName('Spark_SQL_basic_example')\
    #     .config('spark.executor.memory','2g') \
    #     .getOrCreate()
    # # create DataFrame
    # df1 = myspark.read.option('header', True).format('csv').load('test.csv')
    # # df1.printSchema()
    # # transform format
    # df2 = df1.selectExpr(  # 转换字符类型
    #     'cast(name as STRING)',
    #     'cast(age as DOUBLE)',
    #     'cast(gender as STRING)'
    # )
    # # df2.printSchema()
    # df2.show()
    # df2.createOrReplaceTempView('customer')
    # cust1 = myspark.sql('SELECT * FROM customer WHERE age BETWEEN 25 and 30')
    # cust1.limit(2).show()
    # cust2 = myspark.sql("SELECT * FROM customer WHERE gender LIKE 'M'")
    # cust2.show()

    # ml_package()
    # pipeline_example()
    # decision_tree_example_withcrossvali()
    tfidf()
