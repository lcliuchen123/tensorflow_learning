# coding:utf-8

# 评估两个geek_id的相似程度
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType, StructType, StructField, LongType, FloatType, ArrayType
from datetime import datetime, timedelta
import time


@F.udf(returnType=FloatType())
def is_match(s_i, s_j):
    """对比两个字符串是否匹配"""
    if not s_i or not s_j or len(s_i) <= 0 or len(s_j) <= 0:
        return 0
    score = 0
    s_i = s_i.split(',')
    s_j = s_j.split(',')
    jiao = 0
    for ele in s_i:
        if ele in s_j:
            jiao += 1
    score = jiao / (len(s_i) + len(s_j) - jiao)
    return score


def main(spark, path):
    # 读取 & 清除相同的geek_id pair对
    data = spark.read.csv(path, sep='\t').toDF('geek_id_i', 'geek_id_j')
    data = data.filter("geek_id_i != geek_id_j")
    all_num = data.count()
    print("data.count: ", all_num)
    # 获取基本信息
    geek_info = spark.table("arc.geek_expect_info").filter("ds='2023-07-25' and source=0")
    geek_info.persist()
    geek_info_i = geek_info.\
        withColumnRenamed('city_code', 'geek_city_i').\
        withColumnRenamed('position_code', 'geek_position_i').\
        withColumnRenamed('low_salary', 'low_salary_i').\
        withColumnRenamed('high_salary', 'high_salary_i').\
        withColumnRenamed('l2_code', 'l2_code_i').\
        withColumnRenamed('l1_code', 'l1_code_i')
    data_with_i = data.join(geek_info_i, data['geek_id_i'] == geek_info_i['geek_id'], 'left')
    data_with_i = data_with_i.groupBy('geek_id_i', 'geek_id_j').agg(
        F.concat_ws(',', F.collect_set('geek_city_i')).alias('geek_city_i'),
        F.concat_ws(',', F.collect_set('geek_position_i')).alias('geek_position_i'),
        F.concat_ws(',', F.collect_set('l2_code_i')).alias('l2_code_i'),
        F.concat_ws(',', F.collect_set('l1_code_i')).alias('l1_code_i')
    )

    geek_info_j = geek_info. \
        withColumnRenamed('city_code', 'geek_city_j'). \
        withColumnRenamed('position_code', 'geek_position_j'). \
        withColumnRenamed('l2_code', 'l2_code_j'). \
        withColumnRenamed('l1_code', 'l1_code_j')
    data_with_j = data_with_i.join(geek_info_j, data['geek_id_j'] == geek_info_j['geek_id'], 'left')
    columns = ['geek_id_j', 'geek_id_i', 'geek_city_i', 'geek_position_i', 'l2_code_i', 'l1_code_i']
    data_with_j = data_with_j.groupBy(columns).agg(
        F.concat_ws(',', F.collect_set('geek_city_j')).alias('geek_city_j'),
        F.concat_ws(',', F.collect_set('geek_position_j')).alias('geek_position_j'),
        F.concat_ws(',', F.collect_set('l2_code_j')).alias('l2_code_j'),
        F.concat_ws(',', F.collect_set('l1_code_j')).alias('l1_code_j')
    )

    # 计算匹配的cp占比
    match_score = data_with_j.\
        withColumn('city_match_score', is_match('geek_city_i', 'geek_city_j')).\
        withColumn('position_match_score', is_match('geek_position_i', 'geek_position_j')).\
        withColumn('l2_match_score', is_match('l2_code_i', 'l2_code_j')).\
        withColumn('l1_match_score', is_match('l1_code_i', 'l1_code_j'))
    match_score = match_score.select(
        (F.sum('city_match_score') / all_num).alias('city_match_score'),
        (F.sum('position_match_score') / all_num).alias('position_match_score'),
        (F.sum('l2_match_score') / all_num).alias('l2_match_score'),
        (F.sum('l1_match_score') / all_num).alias('l1_match_score')
    )
    match_score.show(10, False)


if __name__ == "__main__":
    spark = SparkSession.builder.master('yarn').\
        enableHiveSupport().\
        config("hive.exec.dynamic.partition", "true").\
        config("hive.exec.dynamic.partition.mode", "nonstrict").\
        getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    path = "hdfs://bzl-hdfs/user/arc_seventeen/liuchen04/eges_c/answer/2023-07-25/pairs_7"
    main(spark, path)
    print("Finished!")
    spark.stop()
