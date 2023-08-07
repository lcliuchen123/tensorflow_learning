# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType, StructType, StructField, LongType, FloatType, ArrayType
from datetime import datetime, timedelta
import time
import argparse


# 获取样本数据
def get_n_days_before(x: str, n_days: int) -> str:
    x_dt = datetime.strptime(x, "%Y-%m-%d")
    ret_dt = x_dt - timedelta(days=n_days)
    return ret_dt.strftime("%Y-%m-%d")


def main(spark, table_name, start_date, end_date, graph_path, side_info_path, node_encode_path):
    # 获取用户点击序列，构造图
    sql = f"""
       select * from {table_name}
       where ds >='{start_date}' and ds <= '{end_date}'
       and is_click = 1 and flag = 1
    """
    my_table = spark.sql(sql)
    my_table.persist()
    print("my_table.count: ", my_table.count())
    a = my_table.select('ds', 'geek_id', 'form_id', 'list_time')
    print("form_id.count: ", a.select('form_id').distinct().count())
    userWindow = Window.partitionBy("form_id").orderBy("list_time")
    # 得到上一次的 开聊时间和开聊job_id (F.lag)
    b = a.withColumn("geek_id_last", F.lag("geek_id", 1).over(userWindow))
    c = b.filter('geek_id_last is not null')
    d = c.groupBy('geek_id', 'geek_id_last').agg(F.count('*').alias('weight'))
    # 生成pair—key，为的是合并a-b和b-a
    e = d.withColumn("pair_key", F.concat_ws('_', 'geek_id', 'geek_id_last'))
    f = e.withColumn("pairs", F.sort_array(F.split(e["pair_key"], "_")))
    g = f.groupBy('pairs').agg(F.sum('weight').alias('total_weight'))
    h = g.withColumn("pair_a", g['pairs'][0]).withColumn("pair_b", g['pairs'][1]).drop('pairs')
    i = h.orderBy(F.desc('total_weight'))
    g = i[['pair_a', 'pair_b', 'total_weight']]
    print("g.count: ", g.count())
    # 统计权重的分布情况
    weight_percent = g.approxQuantile('total_weight', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 1e-5)
    print(f"weight_percent: {weight_percent}")
    g.repartition(5).write.mode("overwrite").option("delimiter", " ").format("csv").save(graph_path)
    print('保存有权图graph结束')

    # 获取每个节点的side info（特征信息）
    geek_numeric_features = [
        'geek_gender', 'geek_rev_age', 'geek_school_degree', 'geek_workyears',
        'five_basic_is_overseas', 'geek_apply_status', 'five_get_geek_high_salary',
        'five_get_geek_low_salary', 'five_geek_complete_time_diff_day',
        'five_get_geek_list_num_2d', 'five_get_geek_click_num_2d',
        'five_get_geek_click_rate_2d', 'five_get_geek_list_num_7d',
        'five_get_geek_click_num_7d', 'five_get_geek_click_rate_7d',
        'five_get_geek_list_num_30d', 'five_get_geek_click_num_30d',
        'five_get_geek_click_rate_30d'
    ]
    geek_array_feature = ['five_get_geek_city_set', 'five_get_geek_position_set',
                          'five_get_geek_l2code_set', 'five_get_geek_l1code_set',
                          'five_get_geek_click_recent30']
    feature_list = geek_numeric_features + geek_array_feature
    # 最近的一条样本的特征, 防止特征穿越
    aa = my_table.select('geek_id', *feature_list,
                         F.row_number().over(Window.partitionBy('geek_id').orderBy(F.asc('list_time'))).alias('rn'))
    bb = aa.filter('rn=1').drop('rn')
    print("bb.count: ", bb.count())
    bb.repartition(5).write.option("header", "true").mode("overwrite").format("csv").save(side_info_path)
    print('保存side_info结束')

    # 对geek_id进行编码, node_cnt: 节点在样本中的出现次数; node_index: 按照node_cnt排序进行编码
    aaa = my_table.groupBy('geek_id').agg(F.count('*').alias('node_cnt'))
    bbb = aaa.withColumn('node_index', (F.row_number().over(Window.orderBy(F.desc('node_cnt')))-1))
    print("bbb.count: ", bbb.count())
    bbb.repartition(1).write.option("header", "true").mode("overwrite").format("csv").save(node_encode_path)
    print('保存encode结束,打完收工')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--table_name", type=str, default=None)
    parser.add_argument("--form_type", type=str, default=None)
    parser.add_argument("--days", type=int, default=29)
    args = parser.parse_args()

    stat_time = time.time()
    # table_name = 'dm_boss_biz_rec.lc04_get_c_answer_train_table'
    # n_day_before = 6

    table_name = args.table_name
    n_day_before = args.days
    form_type = args.form_type
    today = datetime.today().strftime("%Y-%m-%d")
    end_date = get_n_days_before(today, 1)
    start_date = get_n_days_before(end_date, int(n_day_before))
    print(f'{n_day_before}天数据：{start_date}<= 日期 <={end_date}')
    # start_date='2022-07-30'
    # end_date='2022-07-30'
    # form_type = 'answer'
    graph_location = f"hdfs://bzl-hdfs/user/arc_seventeen/liuchen04/eges_c/{form_type}/{end_date}/graph_{n_day_before + 1}"
    side_info_location = f"hdfs://bzl-hdfs/user/arc_seventeen/liuchen04/eges_c/{form_type}/{end_date}/side_info_{n_day_before + 1}"
    nodeid_encode_location = f"hdfs://bzl-hdfs/user/arc_seventeen/liuchen04/eges_c/{form_type}/{end_date}/nodeid_encode_{n_day_before + 1}"
    spark = SparkSession.builder.appName("liuchen04_eges").\
        config("spark.sql.sources.partitionOverwriteMode", "dynamic").\
        config("hive.exec.dynamic.partition.mode", "nonstrict").\
        config("spark.driver.maxResultSize", "2g").\
        config("spark.rpc.message.maxSize", 555).\
        config("spark.network.timeout", 300000).\
        enableHiveSupport().\
        getOrCreate()
    main(spark, table_name, start_date, end_date, graph_location, side_info_location, nodeid_encode_location)
    spark.stop()
    end_time = time.time()
    cost_time = (end_time - stat_time) / 60
    print(f"Finished ! cost time is: {cost_time} min")

