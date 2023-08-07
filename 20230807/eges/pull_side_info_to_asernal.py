# -*- coding: utf-8 -*-
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", type=str, required=True)
    parser.add_argument("-output", type=str, required=True)
    parser.add_argument("-format", type=str, required=False, default='parquet')
    parser.add_argument("-header", type=str, required=False, default="False")
    args = parser.parse_args()
    spark = SparkSession \
        .builder \
        .appName("liuchen04_side_info") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .config("spark.default.parallelism", 1000) \
        .config("spark.executor.memoryOverhead", "4g") \
        .config("spark.driver.memoryOverhead", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .enableHiveSupport() \
        .getOrCreate()
    print('拉取side_info,')
    print('是否表头,', args.header)
    input_data = spark.read.format("csv").option("header", args.header).load(args.input)
    input_data.show()
    input_data.coalesce(1).write.format("csv").option("header", args.header).option("delimiter", ",").mode("overwrite").save(args.output)
    spark.stop()
