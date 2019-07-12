# !python3
# coding=utf-8
"""
-------------------------------------------------
   File Name：helloworld
   Description :
   Author : 小巫女
   date：2019/6/13
-------------------------------------------------
   Change Activity:
   2019/6/13 15:44
-------------------------------------------------
"""
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    print("Successfully imported Spark Modules")
except ImportError as e:
    print("Can not import Spark Modules", e)

