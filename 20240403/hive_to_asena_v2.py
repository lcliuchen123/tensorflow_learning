# coding:utf-8

# 加载spark 增加文本特征
import os
import sys
from os.path import abspath, dirname
current_path = abspath(dirname(__file__))
sys.path.append("{}/../..".format(current_path))
sys.path.append(".")
sys.path.append("..")
import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F

seq_feature_length_dict = {
    'fea_name': 30,
}

fea_type_to_spark_type = {
    'int': LongType(),
    'long': LongType(),
    'float': DoubleType(),
    'double': DoubleType(),
    'int-vector': LongType(),
    'long-vector': LongType(),
    'float-vector': DoubleType()
}


# 输出是LongType
@F.udf(returnType=ArrayType(elementType=LongType(), containsNull=False))
def parse_vector_str(s, length, fea_type):
    """
       字符串序列特征转化为int或者float
    """
    if isinstance(s, str):
        s = s.replace('[', '').replace(']', '')
        res = s.split(",")
    elif isinstance(s, list):
        res = s
    else:
        print("Input must list type or str type")
        return None
    print("res: ", res)
    if isinstance(res, list):
        for i in range(len(res)):
            try:
                if fea_type in ('int-vector', 'long-vector'):
                    res[i] = int(float(res[i]))
                elif fea_type == 'float-vector':
                    res[i] = float(res[i])
            except:
                res[i] = -3
    if fea_type in ('int-vector', 'long-vector'):
        if res is None or len(res) <= 0:
            return [-3] * length
        else:
            return res[:length] + [-3] * (length - len(res))
    elif fea_type == 'float-vector':
        if res is None or len(res) <= 0:
            return [-3.0] * length
        else:
            return res[:length] + [-3.0] * (length - len(res))
    else:
        print("Not int or float")
        return [-3] * length


# 输出是FloatType
@F.udf(returnType=ArrayType(elementType=FloatType(), containsNull=False))
def parse_vector_float(s, length, fea_type):
    """
       字符串序列特征转化为int或者float
    """
    if isinstance(s, str):
        s = s.replace('[', '').replace(']', '')
        res = s.split(",")
    elif isinstance(s, list):
        res = s
    else:
        print("Input must list type or str type")
        return [0.0] * length
    print("res: ", res)
    if isinstance(res, list):
        for i in range(len(res)):
            try:
                if fea_type in ('int-vector', 'long-vector'):
                    res[i] = int(float(res[i]))
                elif fea_type == 'float-vector':
                    res[i] = float(res[i])
            except:
                res[i] = -3
    if fea_type in ('int-vector', 'long-vector'):
        if res is None or len(res) <= 0:
            return [-3] * length
        else:
            return res[:length] + [-3] * (length - len(res))
    elif fea_type == 'float-vector':
        if res is None or len(res) <= 0:
            return [-3.0] * length
        else:
            return res[:length] + [-3.0] * (length - len(res))
    else:
        print("Not int or float")
        return [-3] * length


def cover_data_type(data, f1_columns_dict, new_columns_dict, text_columns_dict):
    """
        转化data的数据类型，确保与模型输入一致
        处理空值
    """
    if not data:
        return
    all_columns = data.columns
    for fea_name in all_columns:
        # 先获取特征类型
        fea_type = None
        if fea_name in f1_columns_dict:
            fea_type = f1_columns_dict[fea_name]
        elif fea_name in new_columns_dict:
            fea_type = new_columns_dict[fea_name]
        elif fea_name in text_columns_dict:
            fea_type = text_columns_dict[fea_name]
        elif fea_name in ['job_id', 'exp_id']:
            fea_type = 'int'
        elif fea_name in ['is_addf', 'is_success']:
            fea_type = 'float'
        else:
            print(f" {fea_name} Not find the feature column type")
            return None
        fea_spark_type = fea_type_to_spark_type[fea_type]
        print(f"fea_name: {fea_name}, fea_type: {fea_type}, fea_spark_type: {fea_spark_type}")
        # 序列特征和非序列特征
        default_value_dic = {'int': 0, 'long': 0, 'float': 0.0, 'double': 0.0}
        try:
            if fea_name not in seq_feature_length_dict.keys():
                data = data.withColumn(fea_name, F.when(data[fea_name].cast(fea_spark_type).isNull(),
                                                        default_value_dic[fea_type]).
                                       otherwise(data[fea_name].cast(fea_spark_type)))
            else:
                length = seq_feature_length_dict[fea_name]
                if fea_name in text_columns_dict:
                    data = data.withColumn(fea_name, parse_vector_float(fea_name, F.lit(length), F.lit(fea_type)))
                else:
                    data = data.withColumn(fea_name, parse_vector_str(fea_name, F.lit(length), F.lit(fea_type)))
        except Exception as e:
            # print("Exception: ", e)
            print("****Error: can not find this col in feature columns: %s " % fea_name)
    return data


def get_day_list(today, data_type, valid_day=['2024-03-11']):
    """获取不同天数的数据"""
    day_list = []
    today_date = datetime.datetime.strptime(today, '%Y-%m-%d')
    print("today_date: ", today_date)
    if data_type == 'valid':
        return valid_day
    day_range = None
    if data_type == 'train':
        day_range = range(2, 5)
    elif data_type == 'test':
        day_range = range(1, 2)
    else:
        print("day_range is null")
        return None
    for i in day_range:
        delta = datetime.timedelta(days=i)
        n_days = today_date - delta
        day = n_days.strftime('%Y-%m-%d')
        day_list.append(day)
    return day_list


def get_features(file_path):
    """获取所有的特征"""
    if not file_path:
        print("file path is none")
        return None
    f1_columns_dict, new_columns_dict, text_columns_dict, cate_columns_dict = {}, {}, {}, {}
    df = pd.read_csv(file_path)
    for i in range(len(df)):
        d_type = df['type'][i].strip()
        fea_name = df['feaName'][i].strip()
        fea_type = df['valueType'][i].strip()
        is_cat = df['is_cate'][i]
        cate_columns_dict[fea_name] = is_cat
        if d_type == 'f1':
            f1_columns_dict[fea_name] = fea_type
        elif d_type == 'new':
            new_columns_dict[fea_name] = fea_type
        elif d_type == 'text':
            text_columns_dict[fea_name] = fea_type
        else:
            print(f"dtype: {d_type} not in f1 new and text")
    num = len(f1_columns_dict) + len(new_columns_dict) + len(text_columns_dict)
    print("get features length is %d" % num)
    return f1_columns_dict, new_columns_dict, text_columns_dict, cate_columns_dict


def hive_hdfs_to_asena(spark, asena_path, today, data_type, file_path):
    """
       today: 今天的日期
       data_type: 标记写入训练集还是测试集还是验证集
    """
    # step1 获取需要选取的列名, 只包含特征
    f1_columns_dict, new_columns_dict, text_columns_dict, cate_columns_dict = get_features(file_path)
    f1_columns = list(f1_columns_dict.keys())
    f1_columns.extend(['job_id', 'exp_id', 'deal_type'])
    new_columns = list(new_columns_dict.keys())
    new_columns.extend(['job_id', 'geek_position'])

    # step2: 获取日期列表，训练集: 最近2～4天的数据；测试集: 最近1天的数据
    day_list = None
    if data_type == 'valid':
        day_list = get_day_list(today, data_type, ['2024-03-11'])
    else:
        day_list = get_day_list(today, data_type)

    if len(day_list) <= 0:
        print("day_list is null")
        return None

    # step3: 获取样本数据
    data = None
    path = ""
    day = ""
    data.repartition(50).write.option('recordType', 'Example'). \
            option('codec', 'org.apache.hadoop.io.compress.GzipCodec'). \
            save(path=path, format='tfrecord', mode='overwrite')
    print("*********End stat %s data************" % day)


def main(spark, path, today, file_path):
    print("=======Load the train data========")
    hive_hdfs_to_asena(spark, path, today, "train", file_path)

    print("=======Load the test data========")
    hive_hdfs_to_asena(spark, path, today, "test", file_path)

    print("=======Load the valid data========")
    hive_hdfs_to_asena(spark, path, today, "valid", file_path)

    print("=======Finished!========")


# spark.default.parallelism=1000
if __name__ == "__main__":
    spark = SparkSession.builder.appName('lc04'). \
        config("spark.default.parallelism", "300"). \
        config("hive.exec.dynamic.partition", "true"). \
        config("hive.exec.dynamic.partition.mode", "nonstrict"). \
        enableHiveSupport(). \
        getOrCreate()
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    file_path = '特征.csv'
    asena_path = sys.argv[1]
    main(spark, asena_path, today, file_path)
    spark.stop()
