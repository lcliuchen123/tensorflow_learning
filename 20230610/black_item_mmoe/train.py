# coding:utf-8

import datetime as dt
import time
import os
import sys

from pirp.model import MMoEModel, get_model_conf
from pirp.utils import TrainingHistory, UpdateHistory


def train_model(version=None):
    # 开始时间
    st = time.time()
    model_conf = get_model_conf(version)
    train_end_dt = dt.datetime.strptime('2024-01-23', '%Y-%m-%d') - dt.timedelta(days=30)
    print("train_end_dt:", train_end_dt)

    train_ds_partitions = ['{0}'.format((train_end_dt - dt.timedelta(days=shift_days)).strftime('%Y-%m-%d')) for
                           shift_days in range(200)]
    test_ds_partitions = ['{0}'.format((train_end_dt + dt.timedelta(days=shift_days)).strftime('%Y-%m-%d')) for
                          shift_days in range(1, 16)]

    print("train_ds_partitions:", train_ds_partitions)
    print("test_ds_partitions:", test_ds_partitions)

    # build datasets
    # train_ds = DataReaderV3(train_ds_partitions, model_conf, batch_size=2048).build_dataset()
    # val_ds = DataReaderV3(test_ds_partitions, model_conf, batch_size=2048).build_dataset()
    # test_ds = DataReaderV3(test_ds_partitions, model_conf, batch_size=32).build_dataset()

    # 该行代码无意义, 上面获取dataset的方式不通用
    train_ds, val_ds, test_ds = [], [], []

    i = 0
    for index, data in enumerate(test_ds):
        if i > 0:
            break
        print(index, data)
        i += 1

    # 处理不同版本
    print('[INFO] Version: {0}'.format(version))
    print('[INFO] Train End Date: {0}'.format(train_end_dt.isoformat()))

    # 直接训练部分
    model = MMoEModel(model_conf, version=version)
    model.build()
    # model.summary()
    model.compile()

    print("log_path: ", model.log_path)
    # 定义callbacks
    callback_log_file = os.path.join(model.log_path, 'callback_log.txt')
    callbacks = [
        TrainingHistory(model.targets, model.log_path, has_val=True),
        UpdateHistory(callback_log_file, train_end_dt.isoformat())
    ]
    model.fit(train_ds, epochs=3, steps_per_epoch=None, validation_data=val_ds, validation_steps=None,
              callbacks=callbacks)
    # 模型训练时间
    runtime_log_file = os.path.join(model.log_path, 'runtime_log.txt')
    fit_et = time.time()

    # 计算auc并评估阈值
    model.evaluate_all(test_ds)
    eval_et = time.time()

    # 修改模型的输出并保存模型
    print("history_model_path: ", model.history_model_path)
    print("current_model_path: ", model.current_model_path)
    model.package_model()
    model.save()
    save_et = time.time()

    # 打印时间
    with open(runtime_log_file, 'w') as f:
        f.write('[Fit Time] {0} min\n'.format((fit_et - st) / 60))
        f.write('[Eval Time] {0} min\n'.format((eval_et - fit_et) / 60))
        f.write('[Save Time] {0} min\n'.format((save_et - eval_et) / 60))


def str_to_dt(dt_str):
    return dt.datetime.strptime(dt_str, '%Y-%m-%d').date()


if __name__ == '__main__':
    # 参数数量
    param_cnt = len(sys.argv)
    if param_cnt == 1:
        train_model()
    elif param_cnt == 2:
        train_model(version=sys.argv[1])
    else:
        raise ValueError('Input argument error!')

