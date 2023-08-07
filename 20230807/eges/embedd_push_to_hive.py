# -*- coding: utf-8 -*-

import time
import argparse
import pandas as pd
from pyspark.sql import SparkSession, functions as F


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--emb_save_path", type=str, default="/group/recall/liuchen04/get/output/emb2.csv")
    args = parser.parse_args()

    spark = SparkSession \
        .builder \
        .appName("EGES_embeding_eval_liuchen04") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .config("spark.default.parallelism", 1000) \
        .config("spark.executor.memoryOverhead", "4g") \
        .config("spark.driver.memoryOverhead", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .enableHiveSupport() \
        .getOrCreate()
    today = time.strftime("%Y-%m-%d")
    # 本地测试
    # output_emb_file = '/group/recall/liuchen04/get/output/emb2.csv'
    # data = pd.read_csv(output_emb_file, delimiter='\t')
    data = pd.read_csv(args.emb_save_path, delimiter='\t', header=None)
    print("data.count: ", len(data))

    # 例行任务
    # data = pd.read_csv(args.output_emb_file, delimiter='\t')
    df = spark.createDataFrame(data).toDF('id', 'embedding')
    df = df.withColumn('ds', F.lit(today)).\
        withColumn('model_name', F.lit('AUSLESE_GRCD-TF-RECALL-ARC-SEVENTEEN-EGES-GEEK')).\
        withColumn('form_type', F.lit('geek_answer'))

    embedding_table_name = 'dm_boss_biz_rec.get_grcd_recall_embedding'
    columns = spark.table(embedding_table_name).columns
    df.show()
    df.select(columns).repartition(2).write.mode('overwrite').insertInto(embedding_table_name)
    print('成功保存至', embedding_table_name, '总条数', df.count())
    spark.stop()


