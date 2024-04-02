# -*- coding: UTF-8 -*-

# lightGCN相关代码

import os, time, logging, argparse, shutil
import queue as Queue, threading, pickle
import faiss as fss
from multiprocessing import Process, Pool, Manager
import multiprocessing as mp
import scipy as sc
from scipy import sparse as sp
import numpy as np
import random
import traceback
import torch

os.environ['DGLBACKEND'] = 'tensorflow'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
# os.system("pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html")
# os.system('pip install networkx -i https://pypi.tuna.tsinghua.edu.cn/simple')
import dgl
from dgl import graph, function as dfn
from dgl.nn.tensorflow.conv import GraphConv

# import tensorflow_recommenders_addons as tfra
logger = logging.getLogger("train")
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
gpu_type = tf.config.experimental.get_device_details(physical_gpus[0])['device_name'] if len(
    physical_gpus) > 0 else 'cpu'

# /code/yanmingxiang/package_code/gcn/gcn-dgl/20220817211518/main
# [683646, 3239138, 281916, 473663, 18898520, 599740] 7.5
# [581558, 2718007, 234492, 382873, 16039895, 489052] 7.1
# [510026, 2659296, 241042, 444247, 19408255, 577460] 6.24
# logger.info(f"device info {physical_gpus}, {gpu_type}")
gpu_type = 0 if '3090' in gpu_type.lower() else 1


# if gpus:
#     # 设置两个逻辑GPU模拟多GPU训练
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288),
#              tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)

# with tf.device('/device:GPU:1'):
#    x = tf.Variable(tf.keras.initializers.RandomUniform(-.01,.01)( shape=(10,10)) )
def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-task", type=str, required=False, default='train')
    parser.add_argument("-targe_city", type=str, required=False)
    parser.add_argument("-bdir", type=str, required=False, default='/dataset/yanmingxiang/gcn')
    parser.add_argument("-input", type=str, required=False, default='city')
    parser.add_argument("-similar", type=str, required=False, default='similar')
    parser.add_argument("-npz_dir", type=str, required=False, default='npz')
    parser.add_argument("-recall", type=str, required=False, default='recall')
    parser.add_argument("-embedding", type=str, required=False, default='embed')
    parser.add_argument("-neg_ration", type=float, required=False, default=0.56)
    # 必须比boss过滤的小,数据处理时无逻辑处理大的情况,当前boss频率阈值为5
    parser.add_argument("-geek_threshold", type=int, required=False, default=5)
    parser.add_argument("-top_n", type=int, required=False, default=1200)
    # parser.add_argument("-denoise", type=float, required=False, default=0.02)
    # parser.add_argument("-dropedge", type=float, required=False, default=0.1)
    args, unknown = parser.parse_known_args()
    return args


args = parse_args()


def get_files_dir(path, return_size=True, re=False):
    data = []
    if not os.path.exists(path):
        return data
    if not os.path.isdir(path):
        data.append((path, os.path.getsize(path)))
        return data
    for filename in os.listdir(path):
        if filename.startswith('.'): continue  # 过滤隐藏文件
        if filename.startswith('_'): continue  # 过滤状态文件
        path1 = os.path.join(path, filename)
        if os.path.isdir(path1) and re:
            data.extend(getFileInPath(path1))
        elif os.path.isdir(path1):
            continue  # 否则过滤目录
        else:
            data.append((path1, os.path.getsize(path1)))
    data = [(v, s) for v, s in data if not (v.startswith('.') or v.startswith('_'))]
    return data


class FullBatchDataset:
    """
    建议将id信息进行紧密映射,稀疏矩阵进行 sum等操作后为密度矩阵
    """

    def __init__(self, adj, bCorpus, gCorpus, bCorpus_mp, gCorpus_mp, boss_position_nums,
                 b_intent):  # 矩阵为row->u,col->i,最好coo,表示邻接矩阵
        self.b_intent = b_intent  # 职位改position会存在不一致
        self.bCorpus, self.gCorpus = bCorpus, gCorpus  ##用于存储(jobid,position等详细信息)
        self.bCorpus_mp, self.gCorpus_mp = bCorpus_mp, gCorpus_mp  # 用户存储jobid,expid的->id的map,存储
        self.boss_position_nums = boss_position_nums
        self.n_boss, self.n_geek = len(bCorpus), len(gCorpus)
        # self.w_adj = adj
        # self.adj = (adj>0).astype(np.float32) #存储
        self.adj = adj
        freq = np.array(self.adj.sum(1))
        freq = np.where(freq <= 0.0, 1.0, freq)
        self.freq = 1.0 / np.sqrt(freq)  # 节点权重

        geek_weight = np.array((self.adj > 0).sum(1)).reshape(-1)[self.n_boss:] ** 0.5
        geek_weight = geek_weight / np.sum(geek_weight)  # 抽样概率
        self.geek_weight = np.log(geek_weight).astype(np.float32)
        self.g = dgl.from_scipy(self.adj)
        self.run = False

    def get_next_batch(self, batch_size=1e7, neg_sample=2048):
        queue = Queue.Queue(50)
        self.run = True

        def next_batch():
            edges = np.split(tf.stack(self.g.all_edges()).numpy().T, 2)[0]  # 头半部
            edges_boss = edges[:, 0]
            edges_geek = edges[:, 1]
            size = edges_geek.shape[0]
            indexs = np.arange(size)
            while True:
                if self.run == False:
                    queue.put(None, block=True)
                    break
                np.random.shuffle(indexs)
                for i in range(size // batch_size + 1):
                    if self.run == False: break
                    if (size - batch_size * i < batch_size * 0.01): continue
                    select_ids = indexs[i * batch_size:(i + 1) * batch_size]
                    if select_ids.shape[0] < neg_sample: continue
                    eff_length = (select_ids.shape[0] // neg_sample) * neg_sample
                    select_ids = select_ids[:eff_length]
                    select_boss = edges_boss[select_ids]
                    select_geek = edges_geek[select_ids]
                    select_geek_weight = self.geek_weight[select_geek - self.n_boss]
                    unique_id = np.unique(np.concatenate([select_boss, select_geek]))
                    queue.put((None, select_boss, select_geek, unique_id, select_geek_weight, False), block=True)

        self.thread = threading.Thread(target=next_batch)
        self.thread.start()
        while True:
            yield queue.get(block=True)

    def position_graph_deepwork(self, boss_position_nums, ration):
        ration = min(ration, 0.98)
        keys = np.array(list(boss_position_nums.keys()))
        values = np.array(list(boss_position_nums.values()))
        weights = values.sum()
        select_position, select_weight = [], 0.0
        while True:
            select = np.random.choice(keys, size=256, replace=False, p=values / weights)
            for s in select:
                if s in select_position: continue
                select_weight += boss_position_nums[s] / weights
                if select_weight >= min(ration * 1.1, 0.98): return select_position
                select_position.append(s)

    def get_next_subgraph(self, batch_size=1e7, neg_sample=2048, sub_ration=0.8):
        freq = np.array(self.adj.sum(1))[:self.n_boss].reshape(-1)
        positions = self.bCorpus[:, 2]
        boss_position_nums = {}
        for i, p in enumerate(positions):
            if p in boss_position_nums:
                boss_position_nums[p] = boss_position_nums[p] + freq[i]
            else:
                boss_position_nums[p] = freq[i]

        queue = Queue.Queue(100)

        def next_batch():
            while True:
                if self.run == False:
                    queue.put(None, block=True)
                    break
                choice_position = self.position_graph_deepwork(boss_position_nums, sub_ration)
                select_bossid = np.where(np.isin(self.bCorpus[:, 2], choice_position))[0]  # index
                edges_boss = np.array([i for i in select_bossid
                                       for _ in range(self.adj.indptr[i], self.adj.indptr[i + 1])])  # 段对应,存在为0的则偏移
                edges_geek = np.array([self.adj.indices[j] for i in select_bossid
                                       for j in range(self.adj.indptr[i], self.adj.indptr[i + 1])])  # 列,对应
                subgraph_node = np.unique(np.concatenate([edges_boss, edges_geek]))
                subgraph = self.g.subgraph(subgraph_node)

                edges = np.split(tf.stack(subgraph.all_edges()).numpy().T, 2)[0]
                edges_boss = edges[:, 0]
                edges_geek = edges[:, 1]
                size = edges_geek.shape[0]
                indexs = np.arange(size)
                np.random.shuffle(indexs)
                for i in range(size // batch_size + 1):
                    if self.run == False: break
                    if size - batch_size * i < batch_size * 0.01: continue
                    select_ids = indexs[i * batch_size:(i + 1) * batch_size]
                    eff_length = (select_ids.shape[0] // neg_sample) * neg_sample
                    select_ids = select_ids[:eff_length]
                    select_boss = edges_boss[select_ids]
                    select_geek = edges_geek[select_ids]
                    unique_id = np.unique(np.concatenate([select_boss, select_geek]))
                    select_geek_weight = self.geek_weight[subgraph.ndata[dgl.NID].numpy()[select_geek] - self.n_boss]
                    queue.put((subgraph_node, select_boss, select_geek, unique_id, select_geek_weight, i == 0),
                              block=True)

        self.thread = threading.Thread(target=next_batch)
        self.run = True
        self.thread.start()
        while True:
            yield queue.get(block=True, timeout=16)

    def terminate_iter(self, iters):
        if not self.run: return
        self.run = False
        for i in range(200):
            try:
                dt = next(iters)
                if dt is None: return
            except queue.Empty:
                print("block empty")
                return
        print("conn't end iters")

    @classmethod
    def create_from_file(cls, train_file, clear_low_freque=True):
        bCorpus, gCorpus = [], []  ##用于存储(jobid,position等详细id未映射信息)
        bCorpus_mp, gCorpus_mp = {}, {}  # 用户存储jobid,expid的->id的map,存储
        b_intent = {}
        n_boss, n_geek = 0, 0
        train_data = []  # 存储的是交互的投影后的id信息
        # 进行id投影化
        boss_position_nums = {}
        with open(train_file) as train:  # 文件格式,(city,job_id,boss_id,position_id,[(exp_id,geek_id,position_id)]
            for line in train:
                city, job_id, boss_id, boss_position, gid, nums, intent = line.strip().split('\x01')
                jobid, bossid, posid = map(int, [job_id, boss_id, boss_position])
                bCorpus_mp[jobid] = n_boss
                b_intent[jobid] = [int(v) for v in intent.split(',')]
                n_boss += 1
                bCorpus.append((jobid, bossid, posid))
                geeks = [g.split('\x03') for g in gid.split('\x02')]
                posg = []
                for e, g, p, actime, _ in geeks:
                    # actime = int(actime/60000)#分钟
                    e, g, p = int(e), int(g), int(p)
                    if e not in gCorpus_mp:
                        gCorpus_mp[e] = n_geek
                        gCorpus.append((e, g, p))  # exp,geek,position
                        n_geek += 1  # 计算链接
                    posg.append((gCorpus_mp[e], actime))  # id化后的变化
                train_data.append(posg)  # 交互expid的投影后的id,未加n_boss

        adj = sp.dok_matrix((n_boss + n_geek, n_boss + n_geek),
                            dtype=np.float32)
        for i in range(len(train_data)):  # 将geekid平移 n_boss个
            for j, act in train_data[i]:
                adj[i, n_boss + j] = 1.0  # 防止act 为0
                adj[n_boss + j, i] = 1.0
        adj = adj.tocsr()  # 存储,indptr表示每个行的元素的检索位置,indices表示列
        bCorpus, gCorpus = np.array(bCorpus), np.array(gCorpus)  # 存储
        bCorpus_mp = np.array(list(bCorpus_mp.items()))
        gCorpus_mp = np.array(list(gCorpus_mp.items()))
        boss_position_nums = np.array([(k, v) for k, v in boss_position_nums.items()])
        if clear_low_freque:
            adj, gCorpus, gCorpus_mp = cls.clear_low_frequ_geek(adj, gCorpus, n_boss)
        return cls(adj, bCorpus, gCorpus, bCorpus_mp, gCorpus_mp, boss_position_nums, b_intent)

    @classmethod
    def create_from_data(cls, path, clear_low_freque=False):
        adj = sp.load_npz(os.path.join(path, "adj.npz"))
        corpus = np.load(os.path.join(path, "corpus.npz"))
        bCorpus = corpus['boss']
        gCorpus = corpus['geek']
        bCorpus_mp = corpus['boss_corpus']
        gCorpus_mp = corpus['geek_corpus']
        boss_position_nums = corpus['boss_position_nums']
        with open(os.path.join(path, "intent"), 'rb') as handle:
            intent = pickle.load(handle)
        if clear_low_freque:
            adj, gCorpus, gCorpus_mp = cls.clear_low_frequ_geek(adj, gCorpus, len(bCorpus))
        return cls(adj, bCorpus, gCorpus, bCorpus_mp, gCorpus_mp, boss_position_nums, intent)

    @staticmethod
    def clear_low_frequ_geek(adj, gCorpus, n_boss):
        freq = np.array(adj.sum(1)).reshape(-1)
        index = np.where(freq >= args.geek_threshold)[0]  # 频次过滤,geek
        new_adj = adj[index, :].tocsc()[:, index].tocsr()
        geek_args = index[index >= n_boss] - n_boss
        new_gCorpus = gCorpus[geek_args, :]
        new_gCorpus_mp = np.stack([new_gCorpus[:, 0], np.arange(new_gCorpus.shape[0])], axis=1)
        return new_adj, new_gCorpus, new_gCorpus_mp

    @staticmethod
    def save_data(graph, path):
        if os.path.exists(path):
            os.rmdir(path)
        os.mkdir(path)
        sp.save_npz(os.path.join(path, "adj"), graph.adj)
        np.savez(os.path.join(path, "corpus"), boss=graph.bCorpus, geek=graph.gCorpus,
                 boss_corpus=graph.bCorpus_mp, geek_corpus=graph.gCorpus_mp,
                 boss_position_nums=graph.boss_position_nums
                 )
        with open(os.path.join(path, "intent"), 'wb') as handle:
            pickle.dump(graph.b_intent, handle)


class Recall:
    def __init__(self, embedding, datasets, city_name, top_n=1000):
        self.boss_embedding, self.geek_embedding = np.split(embedding, [datasets.n_boss, ])
        self.datasets = datasets
        self.top_n = top_n
        self.city_name = city_name

    @staticmethod
    def save_embedding(embedding, city_name):
        np.savez(os.path.join(args.bdir, args.embedding, city_name), em=embedding)

    @staticmethod
    def load_recall(city_name):
        embedding = np.load(os.path.join(os.path.join(args.bdir, args.embedding, city_name + ".npz")))['em']
        datasets = FullBatchDataset.create_from_data(os.path.join(os.path.join(args.bdir, args.npz_dir, city_name)),
                                                     clear_low_freque=False)
        return Recall(embedding, datasets, city_name, top_n=args.top_n)

    def recall_save_with_intent_filter(self):
        with open(os.path.join(args.bdir, args.recall, self.city_name), "w") as save:
            for i in range(self.datasets.n_boss):
                boss_info = self.datasets.bCorpus[i]
                b_intent = self.datasets.b_intent[boss_info[0]]
                candicate = np.where(np.isin(self.datasets.gCorpus[:, 2], b_intent))[0]
                if b_intent == -1:
                    candicate = np.arange(self.datasets.n_geek)
                boss_info = "\x01".join(map(str, boss_info))
                visited = set([self.datasets.adj.indices[j] - self.datasets.n_boss
                               for j in range(self.datasets.adj.indptr[i], self.datasets.adj.indptr[i + 1])])
                visited = set(self.datasets.gCorpus[list(visited), 1])
                scores = np.dot(self.geek_embedding[candicate], self.boss_embedding[i])
                topn_k = candicate[np.argsort(-scores)[:self.top_n + 3 * len(visited)]]
                recall_data = [self.datasets.gCorpus[j] for j in topn_k.tolist() if
                               self.datasets.gCorpus[j, 1] not in visited][:self.top_n]
                recall_geek = ','.join(['\x02'.join(map(str, w)) for w in recall_data])
                save.write(boss_info + "\x01" + recall_geek + "\n")

    # 对于头部城市需要拆分成2个子城市进行召回加速召回速度 101280100,101020100,101010100,101280600,101270100,101200100,101180100,101210100
    def recall_save(self, part=0, num_part=1):  # 不限制长度
        # 暴力检索,geek维度的过滤
        index = fss.IndexFlatIP(self.geek_embedding.shape[1])
        index.add(self.geek_embedding)
        with open(os.path.join(args.bdir, args.recall, f"{self.city_name}_{part}"), "w") as save:
            step = self.datasets.n_boss // num_part + 1
            for i in range(part * step, min((part + 1) * step, self.datasets.n_boss)):
                boss_info = self.datasets.bCorpus[i]
                boss_info = "\x01".join(map(str, boss_info))
                visited = set([self.datasets.adj.indices[j] - self.datasets.n_boss
                               for j in range(self.datasets.adj.indptr[i], self.datasets.adj.indptr[i + 1])])
                visited = set(self.datasets.gCorpus[list(visited), 1])
                scores, recall_ids = index.search(self.boss_embedding[i:i + 1], self.top_n + int(len(visited) * 2))
                recall_data = [self.datasets.gCorpus[j] for j in recall_ids.tolist()[0] if
                               self.datasets.gCorpus[j, 1] not in visited][:self.top_n]
                recall_geek = ','.join(['\x02'.join(map(str, w)) for w in recall_data])
                save.write(boss_info + "\x01" + recall_geek + "\n")

    def recall_save_gpu(self):  # 限制长度
        k = np.arange(self.geek_embedding.shape[0])
        v = self.geek_embedding
        feature_data = self.boss_embedding
        v_shape = v.shape[0]
        s_size = 120000
        size_split = [s_size] * (v_shape // s_size) + [v_shape % s_size]
        k_split = tf.split(k, size_split)
        v_split = tf.split(v, size_split)
        topk = 1800
        recalls, recallg, recallsc = [], [], []
        for i in range(feature_data.shape[0] // 2048 + 1):
            input_ids = feature_data[i * 2048:(i + 1) * 2048]
            candication_ids, topk_scores = [], []
            for j in range(len(v_split)):
                k_, v_ = k_split[j], v_split[j]
                logit = tf.einsum('bd,nd->bn', input_ids, v_)
                if k_.shape[0] <= topk:
                    candication_ids.append(tf.tile(tf.reshape(k_, (1, -1)), [input_ids.shape[0], 1]))
                    topk_scores.append(logit)
                    continue
                topk_score, topk_id = tf.math.top_k(logit, k=topk)
                topk_k = tf.gather(k_, topk_id)
                candication_ids.append(topk_k)
                topk_scores.append(topk_score)
            candication_ids = tf.concat(candication_ids, axis=1)
            topk_scores = tf.concat(topk_scores, axis=-1)
            if candication_ids.shape[0] < topk:
                topk_ids = tf.argsort(-topk_scores, axis=-1)[:, :topk]
            else:
                _, topk_ids = tf.math.top_k(topk_scores, k=topk)
            topk_ks = torch.gather(torch.from_numpy(candication_ids.numpy()), 1,
                                   torch.from_numpy(topk_ids.numpy().astype(np.int64))
                                   )  # 排序
            topk_sc = torch.gather(torch.from_numpy(topk_scores.numpy()), 1,
                                   torch.from_numpy(topk_ids.numpy().astype(np.int64))
                                   )  # 排序
            recalls.append(topk_ks.numpy())
            recallsc.append(topk_sc.numpy())
        recalls = np.concatenate(recalls)
        recallsc = np.concatenate(recallsc)
        return recallsc, recalls

    def similar_u_gpu(self):
        k = np.arange(self.datasets.bCorpus.shape[0])
        v = self.boss_embedding
        feature_data = v
        v_shape = v.shape[0]
        s_size = 280000
        size_split = [s_size] * (v_shape // s_size) + [v_shape % s_size]
        k_split = tf.split(k, size_split)
        v_split = tf.split(v, size_split)
        topk = 301
        recalls, recallg, recallsc = [], [], []
        for i in range(feature_data.shape[0] // 2048 + 1):
            input_ids = feature_data[i * 2048:(i + 1) * 2048]
            candication_ids, topk_scores = [], []
            for j in range(len(v_split)):
                k_, v_ = k_split[j], v_split[j]
                logit = tf.einsum('bd,nd->bn', input_ids, v_)
                if k_.shape[0] <= topk:
                    candication_ids.append(tf.tile(tf.reshape(k_, (1, -1)), [input_ids.shape[0], 1]))
                    topk_scores.append(logit)
                    continue
                topk_score, topk_id = tf.math.top_k(logit, k=topk)
                topk_k = tf.gather(k_, topk_id)
                candication_ids.append(topk_k)
                topk_scores.append(topk_score)
            candication_ids = tf.concat(candication_ids, axis=1)
            topk_scores = tf.concat(topk_scores, axis=-1)
            if candication_ids.shape[0] < topk:
                topk_ids = tf.argsort(-topk_scores, axis=-1)[:, :topk]
            else:
                _, topk_ids = tf.math.top_k(topk_scores, k=topk)
            topk_ks = torch.gather(torch.from_numpy(candication_ids.numpy()), 1,
                                   torch.from_numpy(topk_ids.numpy().astype(np.int64))
                                   )  # 排序
            topk_sc = torch.gather(torch.from_numpy(topk_scores.numpy()), 1,
                                   torch.from_numpy(topk_ids.numpy().astype(np.int64))
                                   )  # 排序
            recalls.append(topk_ks.numpy())
            recallsc.append(topk_sc.numpy())
        recalls = np.concatenate(recalls)
        recallsc = np.concatenate(recallsc)
        np.savez(f"{args.bdir}/similarnpz/{self.city_name}_u", rids=recalls[:, 1:], rsc=recallsc[:, 1:],
                 binfo=self.datasets.bCorpus)

    @staticmethod
    def save_similar_u(city_name):
        infoz = np.load(f"{args.bdir}/similarnpz/{city_name}_u.npz")
        recalls, recallsc, bs = infoz['rids'], infoz['rsc'], infoz['binfo']
        sc = recallsc
        rids = bs[recalls]
        with open(os.path.join(args.bdir, f"{args.similar}_u", city_name), 'w') as handle:
            for b, s, r in zip(bs, sc, rids):
                # head = f"{b[0]},{b[1]}"
                head = f"{b[0]}\x01{b[1]}"
                tail = ",".join([f"{r[i, 0]}_{r[i, 1]}_{s[i]:.5f}" for i in range(len(s))])
                handle.write(str(city_name) + '\x01' + head + '\x01' + tail + '\n')
        logger.info(f"{city_name=} u2u had saved")

    def similar_i_gpu(self):
        k = np.arange(self.datasets.gCorpus.shape[0])
        v = self.geek_embedding
        feature_data = v
        v_shape = v.shape[0]
        s_size = 280000
        size_split = [s_size] * (v_shape // s_size) + [v_shape % s_size]
        k_split = tf.split(k, size_split)
        v_split = tf.split(v, size_split)
        topk = 301
        recalls, recallg, recallsc = [], [], []
        for i in range(feature_data.shape[0] // 2048 + 1):
            input_ids = feature_data[i * 2048:(i + 1) * 2048]
            candication_ids, topk_scores = [], []
            for j in range(len(v_split)):
                k_, v_ = k_split[j], v_split[j]
                logit = tf.einsum('bd,nd->bn', input_ids, v_)
                if k_.shape[0] <= topk:
                    candication_ids.append(tf.tile(tf.reshape(k_, (1, -1)), [input_ids.shape[0], 1]))
                    topk_scores.append(logit)
                    continue
                topk_score, topk_id = tf.math.top_k(logit, k=topk)
                topk_k = tf.gather(k_, topk_id)
                candication_ids.append(topk_k)
                topk_scores.append(topk_score)
            candication_ids = tf.concat(candication_ids, axis=1)
            topk_scores = tf.concat(topk_scores, axis=-1)
            if candication_ids.shape[0] < topk:
                topk_ids = tf.argsort(-topk_scores, axis=-1)[:, :topk]
            else:
                _, topk_ids = tf.math.top_k(topk_scores, k=topk)
            topk_ks = torch.gather(torch.from_numpy(candication_ids.numpy()), 1,
                                   torch.from_numpy(topk_ids.numpy().astype(np.int64))
                                   )  # 排序
            topk_sc = torch.gather(torch.from_numpy(topk_scores.numpy()), 1,
                                   torch.from_numpy(topk_ids.numpy().astype(np.int64))
                                   )  # 排序
            recalls.append(topk_ks.numpy())
            recallsc.append(topk_sc.numpy())

        recalls = np.concatenate(recalls)
        recallsc = np.concatenate(recallsc)
        np.savez(f"{args.bdir}/similarnpz/{self.city_name}_i", rids=recalls[:, 1:], rsc=recallsc[:, 1:],
                 binfo=self.datasets.gCorpus)

    @staticmethod
    def save_similar_i(city_name):
        infoz = np.load(f"{args.bdir}/similarnpz/{city_name}_i.npz")
        recalls, recallsc, bs = infoz['rids'], infoz['rsc'], infoz['binfo']
        sc = recallsc
        rids = bs[recalls]
        with open(os.path.join(args.bdir, f"{args.similar}_i", city_name), 'w') as handle:
            for b, s, r in zip(bs, sc, rids):
                # head = f"{b[0]},{b[1]}"
                head = f"{b[0]}\x01{b[1]}"
                tail = ",".join([f"{r[i, 0]}_{r[i, 1]}_{s[i]:.5f}" for i in range(len(s))])
                handle.write(str(city_name) + '\x01' + head + '\x01' + tail + '\n')
        logger.info(f"{city_name=} had i2i saved")

    def similar_u(self):
        index = fss.IndexFlatIP(self.geek_embedding.shape[1])
        index.add(self.boss_embedding)
        bs, sc, rids = [], [], []
        for i in range(len(recall.boss_embedding) // 1024 + 1):
            em = recall.boss_embedding[i * 1024:(i + 1) * 1024]
            scores, recall_ids = index.search(em, 301)
            sc.append(scores[:, 1:])
            rids.append(recall.datasets.bCorpus[recall_ids[:, 1:]])
            bs.append(recall.datasets.bCorpus[i * 1024:(i + 1) * 1024])
        bs = np.concatenate(bs)
        sc = np.concatenate(sc)
        rids = np.concatenate(rids)
        with open(os.path.join(args.bdir, args.similar, self.city_name), 'w') as handle:
            for b, s, r in zip(bs, sc, rids):
                # head = f"{b[0]},{b[1]}"
                head = f"{b[0]}"
                tail = ",".join([f"{r[i, 0]}_{r[i, 1]}_{s[i]:.5f}" for i in range(len(s))])
                handle.write(str(self.city_name) + '\x01' + head + '\x01' + tail + '\n')


# 效果不显著,费时
class DropEdge():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, g):
        # Fast path
        if self.p == 0:
            return g
        samples = tf.random.uniform((g.num_edges(),), 0.0, 1.0)
        save_edges = g.edges(form='eid')[tf.cast(samples >= self.p, tf.bool)]
        g = dgl.edge_subgraph(g, save_edges, relabel_nodes=False, store_ids=False)
        return g


class LightGCN(keras.Model):
    def __init__(self, job_vol, exp_vol, dim):
        super(LightGCN, self).__init__()
        # self.edge_dropot = DropEdge(args.dropedge)
        self.job_size = job_vol
        self.exp_size = exp_vol
        self.dim = dim
        self.gcn_layers = [GraphConv(dim, dim, norm='both',
                                     allow_zero_in_degree=True,
                                     weight=False, bias=False
                                     )
                           for i in range(3)]

    def build(self, input_shape=None):
        init_op = keras.initializers.RandomUniform(-.01, .01)
        if len(physical_gpus) > 1 or (gpu_type == 1):
            with tf.device('/gpu:0'):
                self.embedding = self.add_weight(
                    shape=(self.job_size + self.exp_size, self.dim),
                    initializer=init_op,
                    name='id_embeddings',
                    regularizer=None,
                    constraint=None,
                    experimental_autocast=False)
        else:
            with tf.device('/cpu:0'):
                self.embedding = self.add_weight(
                    shape=(self.job_size + self.exp_size, self.dim),
                    initializer=init_op,
                    name='id_embeddings',
                    regularizer=None,
                    constraint=None,
                    experimental_autocast=False)
        self.built = True

    def look_up_embedding(self, graph, method='full'):
        if len(physical_gpus) > 1:
            return self.embedding.value()
        if method == 'full':
            return self.embedding.value()
        elif method == 'subgraph':
            sub_nodes = graph.ndata[dgl.NID]
            embedding = tf.nn.embedding_lookup(self.embedding, sub_nodes)
            return embedding
        print("look_up_embedding method error")
        return None

    def one_hop(self, graph):
        embedding = self.look_up_embedding(graph, "full")
        embedding = self.gcn_layers[0](graph, embedding)
        embedding = tf.nn.l2_normalize(embedding, -1)
        left, right = graph.all_edges()
        size = left.shape[0]
        batch_size = 5000000
        if size > batch_size:
            size = [batch_size] * int(size / batch_size) + [size - int(size / batch_size) * batch_size]
            left = tf.split(left, size)
            right = tf.split(right, size)
        else:
            left, right = [left], [right]
        cos = [tf.einsum("bi,bi->b", tf.nn.embedding_lookup(embedding, l), tf.nn.embedding_lookup(embedding, r))
               for l, r in zip(left, right)]
        return tf.concat(cos, axis=-1)

    def call(self, graph, training=False, method='full'):
        embedding = self.look_up_embedding(graph, method)
        out_embedding = embedding
        for layer in self.gcn_layers:
            embedding = layer(graph, embedding)
            out_embedding += embedding
        return out_embedding, embedding

    def train(self, g, train_boss, train_geek, neg_size, unique_id, i_weight, method='full'):  # 边的paire对
        # g为全量的图有序结构,train_boss,train_geek为g中边的节点
        # 如g为子图时,其也是从g中重新排序后的结果
        # 温度参数0.1-0.3 与sample softmax https://arxiv.org/pdf/2201.02327.pdf
        with tf.device('/device:GPU:0'):
            embedding, _ = self(g, training=True, method=method)
            u_em = tf.nn.embedding_lookup(embedding, train_boss)
            i_em = tf.nn.embedding_lookup(embedding, train_geek)
            u_em = tf.nn.l2_normalize(u_em, -1)
            i_em = tf.nn.l2_normalize(i_em, -1)
            u_em = tf.reshape(u_em, (-1, neg_size, self.dim))
            i_em = tf.reshape(i_em, (-1, neg_size, self.dim))
            # 同行相乘s为其样本,i,j,j为正样本
            logits = tf.einsum("itd,isd->its", u_em, i_em) / 0.06 - i_weight  # 温度参数(n,neg,neg)
            pos_logits = tf.linalg.diag_part(logits)  # (n,neg)
            loss = -tf.reduce_mean(pos_logits)

            # MixGCF下移到logits效果一致
            a = tf.random.uniform(shape=(tf.shape(logits)[0], neg_size, 1), maxval=1.0)
            logits = a * tf.expand_dims(pos_logits, -1) + (1 - a) * logits

            loss += tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.exp(logits), axis=-1)
                                               )
                                   )
            l2_loss = tf.nn.l2_loss(self.embedding)
        return loss, loss + l2_loss * 2e-4


def clear_and_build(path):
    if os.path.exists(path):  # 删除分城市
        shutil.rmtree(path)
    os.mkdir(path)


def clear_data():
    clear_and_build(os.path.join(args.bdir, args.npz_dir))  # 删除分城市
    clear_and_build(os.path.join(args.bdir, args.recall))  # 删除结果
    clear_and_build(os.path.join(args.bdir, args.embedding))  # 删除结果
    clear_and_build(os.path.join(args.bdir, "similarnpz"))  # 删除u2i,u2u中间结果


def prepare(file_city, npz_dir):
    ts = time.time()
    city_name = os.path.basename(file_city)
    # logger.info(f"begin to prepare {city_name} ")
    dataset = FullBatchDataset.create_from_file(file_city, clear_low_freque=True)
    if dataset.n_geek <= 1500 or len(dataset.adj.data) <= 5000: return "-1"
    logger.info(f"end to prepare {city_name} size is {len(dataset.adj.data)}, costtime:{time.time() - ts}")
    dataset.save_data(dataset, os.path.join(npz_dir, city_name))
    return city_name


# Learning to Denoise Unreliable Interactions forGraph Collaborative Filtering 部分改版
# 目前测试效果1%,不显著
def denoise_graph(graph, model):
    cos = model.one_hop(graph).numpy()
    size = int(cos.shape[0] * args.denoise * 2)
    if size % 2 != 0: size += 1
    denoise_edges = np.argsort(cos)[size:]
    # 需要排序，不然生成的子图被破坏原有的边序
    denoise_graph = dgl.edge_subgraph(graph, np.sort(denoise_edges), relabel_nodes=False, store_ids=True)
    return denoise_graph


def test_denoise(city_name):
    ts = time.time()
    city_path = os.path.join(args.bdir, args.npz_dir, city_name)
    logger.info(f"begint to training {city_name=}")
    datasets = FullBatchDataset.create_from_data(city_path, clear_low_freque=False)
    need_subgraph = not (datasets.n_geek <= 4e6 or len(physical_gpus) > 1 or (gpu_type == 1))
    # dim = int(8*np.log(datasets.adj.indices.shape[0]))
    model = LightGCN(datasets.n_boss, datasets.n_geek, min(dim, 128))
    # 单卡能用空间容量,和图大小4e6个节点,大的切图或者双卡
    batch_size = int(20480 * 18432 * 1.5 * (1 + 0.5 * gpu_type)) + (20480 * 20000 * 2.0 if need_subgraph else 0)
    if datasets.adj.indices.shape[0] > 60000000:
        batch_size = int(batch_size * 0.8)
    elif datasets.adj.indices.shape[0] > 50000000:
        batch_size = int(batch_size * 0.9)
    batch_size = int(batch_size * 0.8)

    neg_size = int(128 + int(datasets.n_geek ** args.neg_ration))
    if datasets.n_geek >= 500000:
        neg_size = 3096
    elif datasets.n_geek >= 300000:
        neg_size = 2048

    batch_size = min(batch_size, datasets.adj.indices.shape[0] // 2 * neg_size)  # 能够处理的样本
    batch_size = (batch_size // neg_size // neg_size) * neg_size

    epoch = min(640 + (datasets.adj.indices.shape[0] // batch_size) * 168, 10240)
    logger.info(
        f"{city_name} train config {batch_size=}, {neg_size=}, {epoch=}, n_boss={datasets.n_boss}, n_geek={datasets.n_geek}, edgs={datasets.adj.indices.shape[0]}, subgraph:{need_subgraph}")
    g = datasets.g.to('/device:GPU:0')
    batch_iters = datasets.get_next_batch(batch_size, neg_size)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-4, epoch // 2,
                                                                     end_learning_rate=1e-5, power=0.5)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    subgraph = g
    nloss = 0.0
    for i in range(epoch):
        ts0 = time.time()
        #             if i == int(epoch*0.7) and epoch>=3096:
        #                 g = denoise_graph(g, model)
        #                 datasets.terminate_iter(batch_iters)
        #                 datasets.g = g.to('cpu:0')
        #                 batch_iters = datasets.get_next_batch(batch_size, neg_size)
        #                 subgraph = g
        nodes, boss_id, geek_id, unique_id, geek_weight, flag = next(batch_iters)  # unique_id放入预处理,节约时间
        ts1 = time.time()
        with tf.GradientTape() as tape:
            loss, t_loss = model.train(subgraph, boss_id, geek_id, neg_size, unique_id,
                                       geek_weight.reshape((-1, 1, neg_size)),
                                       'subgraph' if need_subgraph else 'full'
                                       )
        grads = tape.gradient(t_loss, model.trainable_variables)
        ts2 = time.time()
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 子图耗时
        nloss += loss.numpy()
        if i % 100 == 0:
            logger.info(
                f"epoch {i=} train size:{boss_id.shape[0]} train time:{time.time() - ts1:.3f}, apply time:{time.time() - ts2:.3f}, read+train time:{time.time() - ts0:.3f}, loss:{loss.numpy():.3f}, nloss:{nloss / 100:.3f}")
            nloss = 0
    datasets.terminate_iter(batch_iters)
    ids_out_emb, _ = model(g)
    save_embedding = tf.nn.l2_normalize(ids_out_emb, axis=1).numpy()
    Recall.save_embedding(save_embedding, city_name)
    logger.info(f"train {city_name=} train {epoch=},cost time {time.time() - ts:.3f}")


def train_fun_city(city_name, queue=None):
    erro = False
    try:
        # if True:
        ts = time.time()
        city_path = os.path.join(args.bdir, args.npz_dir, city_name)
        logger.info(f"begint to training {city_name=}")
        datasets = FullBatchDataset.create_from_data(city_path, clear_low_freque=False)
        # need_subgraph = not (datasets.n_geek<=4e6 or len(physical_gpus)>1 or (gpu_type==1))
        need_subgraph = False
        dim = int(8 * np.log(datasets.adj.indices.shape[0]))

        model = LightGCN(datasets.n_boss, datasets.n_geek, min(dim, 128))
        # 单卡能用空间容量,和图大小4e6个节点,大的切图或者双卡
        batch_size = int(20480 * 18432 * 1.0 * (1 + 1.2 * gpu_type)) + (20480 * 20000 * 2.0 if need_subgraph else 0)
        if datasets.adj.indices.shape[0] > 60000000:
            batch_size = int(batch_size * 0.8)
        elif datasets.adj.indices.shape[0] > 50000000:
            batch_size = int(batch_size * 0.9)

        neg_size = int(128 + int(datasets.n_geek ** args.neg_ration))
        if datasets.n_geek >= 500000:
            neg_size = 3096
        elif datasets.n_geek >= 300000:
            neg_size = 2048

        batch_size = min(batch_size, datasets.adj.indices.shape[0] // 2 * neg_size)  # 能够处理的样本
        batch_size = (batch_size // neg_size // neg_size) * neg_size

        epoch = min(640 + (datasets.adj.indices.shape[0] // batch_size) * 168, 10240) // 2
        logger.info(
            f"{city_name} train config {batch_size=}, {neg_size=}, {epoch=}, n_boss={datasets.n_boss}, n_geek={datasets.n_geek}, edgs={datasets.adj.indices.shape[0]}, subgraph:{need_subgraph}")
        batch_iters = datasets.get_next_batch(batch_size, neg_size)
        g = datasets.g.to('/device:GPU:0')
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-4, epoch // 2,
                                                                         end_learning_rate=1e-5, power=0.5)
        optimizer = tf.keras.optimizers.Adam(1e-4)
        subgraph = g
        nloss = 0.0
        # 非top城市降低epoch,降低运行时间
        top_city = ['101280100', '101020100', '101010100', '101280600', '101270100', '101200100', '101180100',
                    '101210100']
        if city_name not in top_city and epoch >= 4096:
            epoch = int((epoch - 4096) * 0.3 + 4096)
        for i in range(epoch):
            ts0 = time.time()
            nodes, boss_id, geek_id, unique_id, geek_weight, flag = next(batch_iters)  # unique_id放入预处理,节约时间
            ts1 = time.time()
            with tf.GradientTape() as tape:
                loss, t_loss = model.train(subgraph, boss_id, geek_id, neg_size, unique_id,
                                           geek_weight.reshape((-1, 1, neg_size)),
                                           'subgraph' if need_subgraph else 'full'
                                           )
            grads = tape.gradient(t_loss, model.trainable_variables)
            ts2 = time.time()
            optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 子图耗时
            nloss += loss.numpy()
            if i % 100 == 0:
                logger.info(
                    f"epoch {i=} train size:{boss_id.shape[0]} train time:{time.time() - ts1:.3f}, apply time:{time.time() - ts2:.3f}, read+train time:{time.time() - ts0:.3f}, loss:{loss.numpy():.3f}, nloss:{nloss / 100:.3f}")
                nloss = 0
        datasets.terminate_iter(batch_iters)
        ids_out_emb, _ = model(g)
        save_embedding = tf.nn.l2_normalize(ids_out_emb, axis=1).numpy()
        Recall.save_embedding(save_embedding, city_name)
        logger.info(f"train {city_name=} train {epoch=},cost time {time.time() - ts:.3f}")
        # datasets.terminate_iter(batch_iters)
    except Exception as e:
        traceback.print_exc()
        logger.info(f"train {city_name=} erro ")
        erro = True
        datasets.terminate_iter(batch_iters)
    finally:
        logger.info(f"terminate_iter")
        return erro


def train_fun(city_path, mp_queue):
    logger.info(f"citys is {city_path}")
    for city_name in city_path:
        erro = train_fun_city(city_name)
        if (mp_queue is not None) and (erro is False):
            mp_queue.put(city_name, block=True)


# 进行u2i推荐存储
def create_recall_subprocee(fork_ctx, mp_queue, has_recall):
    save_process = []
    split_city = {'101280100': 3, '101020100': 3, '101010100': 3, '101280600': 3, '101270100': 3, '101200100': 3,
                  '101180100': 2, '101210100': 2}
    while True:
        city_name = mp_queue.get(block=True)
        if city_name is None: break
        num = split_city.get(city_name, 1)
        for i in range(num):
            p = fork_ctx.Process(target=save_recall, args=(city_name, i, num))
            p.start()
            save_process.append(p)
        has_recall.append(city_name)
    [p.join() for p in save_process]


def debug(city_name):
    if city_name is not None:
        logger.info("begin to debug {city_name=}")
        train_fun_city(city_name)
        logger.info("begin to debug recall {city_name=}")
        save_recall(city_name)
        return
    city_files = sorted(get_files_dir(os.path.join(args.bdir, args.input)), key=lambda x: x[1], reverse=True)
    city_files = {os.path.basename(k): v / 1024 / 1024 for k, v in city_files}  # M
    for k, v in city_files.items():
        if v > 50: continue
        train_fun_city(k)
        save_recall(k)


def save_recall2(fork_ctx):
    city_files = get_files_dir(os.path.join(args.bdir, args.embedding))
    save_process = []
    for city_name, _ in city_files:
        city_name = os.path.basename(city_name)
        city_name = city_name.split('.')[0]
        recall = Recall.load_recall(city_name)
        sc, rids = recall.recall_save_gpu()
        p = fork_ctx.Process(target=recall.u2i_to_files, args=(recall, sc, rids))
        # p = fork_ctx.Process(target=save_recall, args=(city_name,) )
        p.start()
        save_process.append(p)
    [p.join() for p in save_process]


def save_recall(city_name, part=0, num_part=1):
    recall = Recall.load_recall(city_name)
    recall.recall_save(part, num_part)
    # recall.recall_save_with_intent_filter()
    logger.info(f"{city_name=} {part=} {num_part=} had saved")


def save_similarU(fork_ctx):
    clear_and_build(os.path.join(args.bdir, f"{args.similar}_u"))
    files = get_files_dir(os.path.join(args.bdir, args.embedding))
    process = []
    for city, _ in files:
        city_name = os.path.basename(city)
        city_name = city_name.split('.')[0]
        recall = Recall.load_recall(os.path.basename(city_name))
        recall.similar_u_gpu()
        p = fork_ctx.Process(target=Recall.save_similar_u, args=(city_name,))
        p.start()
        process.append(p)
    [p.join() for p in process]


def save_similarI(fork_ctx):
    clear_and_build(os.path.join(args.bdir, f"{args.similar}_i"))
    files = get_files_dir(os.path.join(args.bdir, args.embedding))
    process = []
    for city, _ in files:
        city_name = os.path.basename(city)
        city_name = city_name.split('.')[0]
        recall = Recall.load_recall(os.path.basename(city_name))
        recall.similar_i_gpu()
        p = fork_ctx.Process(target=Recall.save_similar_i, args=(city_name,))
        p.start()
        process.append(p)
    [p.join() for p in process]


def main(sp_ctx, fork_ctx):
    clear_data()
    start_time = time.time()
    city_files = sorted(get_files_dir(os.path.join(args.bdir, args.input)), key=lambda x: x[1], reverse=True)
    # top3城市交互,快速开始模型训练,防止内存骤升运行速度降低甚至kill,前50城shuffle
    head, tail = city_files[:20], city_files[20:]
    random.shuffle(head)
    city_files = head + tail

    pool = fork_ctx.Pool(6)
    data_procee_pool = [pool.apply_async(prepare, (cfile, os.path.join(args.bdir, args.npz_dir)))
                        for cfile, _ in city_files]
    city_files = {os.path.basename(k): v / 1024 / 1024 for k, v in city_files}  # M
    pool.close()

    mp_queue = Manager().Queue(100)
    has_recall = []
    thread = threading.Thread(target=create_recall_subprocee, args=(fork_ctx, mp_queue, has_recall))
    thread.start()
    flags = [True] * len(data_procee_pool)
    train_time = time.time()
    # 获取文件<15M的城市信息,同时累计到一起放一个进程运行,在主进程中完成,节约子进程中初始化gpu时间20s*180
    cache_city, mem_num = [], 0.0
    has_train_city = []
    ts = time.time()
    while any(flags):
        for i, r in enumerate(data_procee_pool):
            if (r.ready() is True) and (flags[i] is True): break
        else:
            continue
        flags[i] = False
        city_name = r.get()
        if city_name == '-1': continue
        if city_files[city_name] <= 50:
            cache_city.append(city_name)
            mem_num += city_files[city_name]
            if mem_num >= 400:  # 500M一组
                city_name = cache_city
                mem_num = 0.0
                cache_city = []
            else:
                continue
        else:
            city_name = [city_name]

        p = sp_ctx.Process(target=train_fun, args=(city_name, mp_queue))
        p.start()
        p.join()
        p.close()
        has_train_city.extend(city_name)
    city_name = cache_city
    if len(city_name) > 0:
        # train_fun(city_name, mp_queue)
        p = sp_ctx.Process(target=train_fun, args=(city_name, mp_queue))
        p.start()
        p.join()
        p.close()
        has_train_city.extend(city_name)
    mp_queue.put(None, block=True)
    logger.info(f"training process end, cost time is :{time.time() - train_time:.2f}")
    thread.join()
    logger.info(f"all process end, cost time is :{time.time() - start_time:.2f}")
    error_city = [c for c in has_train_city if c not in has_recall]
    logger.info("{0} {1} citys recall error".format(','.join(error_city), len(error_city)))

# datasets = FullBatchDataset.create_from_data("/dataset/yanmingxiang/gcn/npz/101190400", clear_low_freque=False)
# g = dgl.from_scipy(datasets.adj, device='/device:GPU:0')
# num_nodes=497201, num_edges=20908184 极限满足,101020100
# 文件大于450M需要放置于双卡运行,采用子图求导时apply_gradients耗时增加10倍,双卡也有利于放置更大的batch


