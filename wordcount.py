# !python3
# coding=utf-8
"""
-------------------------------------------------
   File Name：wordcount
   Description :
   Author : 小巫女
   date：2019/6/13
-------------------------------------------------
   Change Activity:
   2019/6/13 16:10
-------------------------------------------------
"""

from __future__ import print_function

import os
import sys

# Path for spark source folder
os.environ['JAVA_HOME'] = 'C:\Program Files\Java\jdk1.8.0_181'
os.environ['SPARK_HOME'] = "D:\\bigdata\\spark"

# Append pyspark to Python Path
sys.path.append("D:\\bigdata\\spark\\python")
sys.path.append("D:\\bigdata\\spark\\python\\lib\\py4j-0.10.4-src.zip")


from pyspark import SparkContext

if __name__ == '__main__':

    if len( sys.argv ) != 3:

        print ('Usage: python input_name output_name')

        exit(1)

    inputFile = sys.argv[1]

    outputFile = sys.argv[2]

    sc = SparkContext()

    text_file = sc.textFile(inputFile)


    counts = text_file.flatMap(lambda line: line.split(' ')).map(lambda word: (word, 1)).reduceByKey(

        lambda a, b: a + b)

    counts.saveAsTextFile(outputFile)


