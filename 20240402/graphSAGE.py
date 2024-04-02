# coding:utf-8

# 图神经网络一般都用dgl库, 该库的demo版本一般都是pytorch

import torch
import dgl
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 定义图神经网络
from dgl.nn.pytorch import SAGEConv


class GraphSAGE(nn.Module):
    """定义模型结构"""
    def __init__(self, int_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layer = nn.ModuleList()
        self.layer.append(SAGEConv(int_feats, n_hidden, aggregator))
        for i in range(1, n_layers - 1):
            print("Enter the layer")
            self.layer.append(SAGEConv(n_hidden, n_hidden, aggregator))
        self.layer.append(SAGEConv(n_hidden, n_classes, aggregator))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        print("self.layer: ", self.layer)

    def forward(self, blocks, feas):
        h = feas
        for i, (layer, block) in enumerate(zip(self.layer, blocks)):
            h = layer(block, h)
            if i != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, my_net, val_nid, batch_s, num_worker, device):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
        dataloader = dgl.dataloading.DataLoader(
            my_net,
            val_nid,
            sampler,
            batch_size=batch_s,
            shuffle=True,
            drop_last=False,
            num_workers=num_worker
        )
        # print("my_net.num_nodes: ", my_net.num_nodes())
        ret = torch.zeros(my_net.num_nodes(), self.n_classes)
        for input_nodes, output_nodes, blocks in dataloader:
            h = blocks[0].srcdata['features'].to(device)
            print("input: ", h.shape)
            for i, (layer, block) in enumerate(zip(self.layer, blocks)):
                block = block.int().to(device)
                h = layer(block, h)
                print("h_1: ", h.shape)
                if i != self.n_layers - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                    print("h_2: ", h.shape)
            ret[output_nodes] = h.cpu()
        return ret


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def evaluate(model, my_net, labels, val_nid, val_mask, batch_s, num_worker, device):
    """评估"""
    model.eval()
    with torch.no_grad():
        label_pred = model.inference(my_net, val_nid, batch_s, num_worker, device)
        print("label_pred: ", len(label_pred.numpy()[0]))
        model.train()
        return (torch.argmax(label_pred[val_mask], dim=1) == labels[val_mask]).float().sum() / len(label_pred[val_mask])


def train_val_split(node_fea):
    """切分训练集，测试集和验证集"""
    train_node_ids = np.array(node_fea.groupby('label_number').
                              apply(lambda x: x.sort_values('node_id_number')['node_id_number'].values[:20]))
    val_node_ids = np.array(node_fea.groupby('label_number').
                              apply(lambda x: x.sort_values('node_id_number')['node_id_number'].values[21:110]))
    test_node_ids = np.array(node_fea.groupby('label_number').
                            apply(lambda x: x.sort_values('node_id_number')['node_id_number'].values[111:300]))
    train_nid = []
    val_nid = []
    test_nid = []
    for (train_nodes, val_nodes, test_nodes) in zip(train_node_ids, val_node_ids, test_node_ids):
        train_nid.extend(train_nodes)
        val_nid.extend(val_nodes)
        test_nid.extend(test_nodes)

    train_mask = node_fea['node_id_number'].apply(lambda x: x in train_nid)
    val_mask = node_fea['node_id_number'].apply(lambda x: x in val_nid)
    test_mask = node_fea['node_id_number'].apply(lambda x: x in test_nid)

    # print(train_mask)
    # print(train_nid)
    print("train_nid_length: ", len(train_nid))
    print("test_nid_length: ", len(test_nid))
    print("valid_nid_length: ", len(val_nid))
    return train_mask, test_mask, val_mask, train_nid, test_nid, val_nid


def loaddata():
    """加载数据"""
    node_fea = pd.read_table('/Users/admin/Desktop/work_notes/B端相似桶/cf_recall_filter/GraphSAGE/cora/cora.content', header=None)
    edges = pd.read_table('/Users/admin/Desktop/work_notes/B端相似桶/cf_recall_filter/GraphSAGE/cora/cora.cites', header=None)
    # 0是node_id, 1434是node label
    node_fea.rename(columns={0: 'node_id', 1434: "label"}, inplace=True)
    nodeID_number_dict = dict(zip(node_fea['node_id'].unique(), range(node_fea['node_id'].nunique())))
    node_fea['node_id_number'] = node_fea['node_id'].map(nodeID_number_dict)
    edges['edge1'] = edges[0].map(nodeID_number_dict)
    edges['edge2'] = edges[1].map(nodeID_number_dict)

    label_dict = dict(zip(node_fea['label'].unique(), range(node_fea['label'].nunique())))
    node_fea['label_number'] = node_fea['label'].map(label_dict)

    src = np.array(edges['edge1'].values)
    dst = np.array(edges['edge2'].values)
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    my_net = dgl.graph((u, v))

    # 这个地方有疑惑，为什么特征值可训练？ 利用节点表示输入
    # 特征列
    fea_id = range(1, 1434)
    tensor_fea = torch.tensor(node_fea[fea_id].values, dtype=torch.float32)
    # 特征权重
    # fea_np = nn.Embedding(2708, 1433)
    # 为什么要更新特征值？按照论文说法应该是不更新feature的
    # fea_np.weight = nn.Parameter(tensor_fea)
    # my_net.ndata['features'] = fea_np.weight

    my_net.ndata['features'] = tensor_fea
    my_net.ndata['label'] = torch.tensor(node_fea['label_number'].values)

    in_feats = 1433
    n_classes = node_fea['label'].nunique()
    print("n_classes: ", n_classes)
    data = in_feats, n_classes, my_net, tensor_fea
    # data = in_feats, n_classes, my_net, fea_np
    train_val_data = train_val_split(node_fea)

    return data, train_val_data


def run(data, train_val_data, args, sample_size, lr, device_num):
    if device_num > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    in_feats, n_classes, my_net, fea_para = data
    hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker = args

    # 设置训练集和测试集
    train_mask, test_mask, val_mask, train_nid, test_nid, val_nid = train_val_data
    nfeat = my_net.ndata['features']
    labels = my_net.ndata['label']
    # 采样
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_size)
    dataloader = dgl.dataloading.DataLoader(
        my_net,
        train_nid,
        sampler,
        batch_size=batch_s,
        shuffle=True,
        drop_last=False,
        num_workers=num_worker
    )

    # num = 0
    # for batch, (input_nodes, output_nodes, block) in enumerate(dataloader):
    # for i in range(100):
    #     input_nodes, output_nodes, blocks = next(iter(dataloader))
    #     num += 1
    #     print("======%d batch=======" % num)
    #     batch_features, batch_label = load_subtensor(nfeat, labels, output_nodes, input_nodes, device)
    #     print("batch_features: ", batch_features)
    #     print("batch_label: ", batch_label)

    model = GraphSAGE(in_feats, hidden_size, n_classes, n_layers, activation, dropout, aggregator)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss_func = nn.CrossEntropyLoss()
    loss_func.to(device)
    for epoch in range(1):
        for batch, (input_nodes, output_nodes, block) in enumerate(dataloader):
            batch_features, batch_label = load_subtensor(nfeat, labels, output_nodes, input_nodes, device)
            # batch_features = block[0].srcdata['features']
            # batch_label = block[-1].dstdata['label']
            # print("batch_features: ", batch_features)
            # print("batch_label: ", batch_label)

            block = [block_.int().to(device) for block_ in block]
            # print("block: ", block)
            model_pred = model(block, batch_features)
            loss = loss_func(model_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1 == 0:
                print("Batch %d | Loss: %.4f" % (batch, loss.item()))

            # 验证模型准确率
            if epoch % 1 == 0:
                val_acc = evaluate(model, my_net, labels, val_nid, val_mask, batch_s, num_worker, device)
                train_acc = evaluate(model, my_net, labels, train_nid, train_mask, batch_s, num_worker, device)
                print("Epoch %d | val acc: %.4f | Train acc: %.4f " % (epoch, val_acc.item(), train_acc.item()))

    acc_test = evaluate(model, my_net, labels, test_nid, test_mask, batch_s, num_worker, device)
    print("Test acc is: %.4f" % acc_test)
    return model
    # return None


if __name__ == "__main__":
    data, train_val_data = loaddata()
    # print(data)
    # print(train_val_data)
    hidden_size = 16
    n_layers = 2
    sample_size = [10, 25]
    activation = F.relu
    dropout = 0.5
    aggregator = 'mean'
    batch_s = 128
    num_worker = 0
    lr = 0.0001
    device = 0
    args = hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker
    trained_model = run(data, train_val_data, args, sample_size, lr, device)








