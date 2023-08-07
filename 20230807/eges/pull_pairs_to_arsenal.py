# -*- coding: utf-8 -*-
import argparse
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, StringType, StructType, StructField, LongType, FloatType, ArrayType


def jFileSystem(sc):
    jFileSystemClass = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    hadoop_configuration = sc._jsc.hadoopConfiguration()
    return jFileSystemClass.get(hadoop_configuration)


def jPath(sc, filepath):
    jPathClass = sc._gateway.jvm.org.apache.hadoop.fs.Path
    return jPathClass(filepath)


# C端的样本没有编码，是在拉取pair的时候进行编码
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pair_input", type=str, required=True)
    parser.add_argument("-label_input", type=str, required=False, default='parquet')
    parser.add_argument("-output", type=str, required=True)
    parser.add_argument("-format", type=str, required=False, default='parquet')

    args = parser.parse_args()
    spark = SparkSession \
        .builder \
        .appName("liuchen04_pairs") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .config("spark.default.parallelism", 100) \
        .config("spark.executor.memoryOverhead", "4g") \
        .config("spark.driver.memoryOverhead", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .enableHiveSupport() \
        .getOrCreate()
    df = spark.read.text(args.pair_input)
    # 按随机数排序
    df2 = df.withColumn('rand', F.rand(seed=42))
    df3 = df2.orderBy(df2.rand)
    df4 = df3.drop(df3.rand)
    df5 = df4.selectExpr(" split(value,'\t')[0] as pairA ", "split(value,'\t')[1] as pairB")
    df6 = df5.filter('pairA!=pairB')  # 剔除x,y相等的情况
    df7 = df6.select(F.col('pairA').cast(LongType()), F.col('pairB').cast(LongType()))
    label_encode_data = spark.read.format("csv").option("header", "true").\
        load(args.label_input).select('geek_id', 'node_index')
    df8 = df7.join(F.broadcast(label_encode_data), df7.pairB == label_encode_data.geek_id, 'left')
    print('写入pairs')
    df8.repartition(20).select('pairA', 'node_index').write.option("header", "false").mode("overwrite").format(
        "csv").save(args.output)
    print('写入pairs结束')
    spark.stop()
