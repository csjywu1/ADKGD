import numpy as np
import random
import torch
import math
from random import shuffle


class Reader:
    def __init__(self, args, path):

        self.ent2id = dict()
        self.rel2id = dict()
        self.id2ent = dict()
        self.id2rel = dict()
        # 一个字典，头实体到尾实体的映射，表示从一个实体（头部）可以到达的所有实体（尾部）
        self.h2t = {}
        # 一个字典，尾实体到头实体的映射，表示可以到达一个实体（尾部）的所有实体（头部）
        self.t2h = {}

        # 记录异常数据的数量
        self.num_anomalies = 0
        # 存储所有的三元组（实体-关系-实体）
        self.triples = []
        # 用于批处理的起始索引
        self.start_batch = 0
        self.path = path

        self.A = {}
        #读取所有的数据，从train.txt， valid.txt, test.txt
        self.read_triples()
        # if self.path == args.data_dir_YAGO or self.path == args.data_dir_NELL or self.path == args.data_dir_DBPEDIA:
        #     self.read_triples_yago3()
        # else:
        #     self.read_triples()
        # 存储原始三元组的集合，用于快速检查是否存在某个特定的三元组
        self.triple_ori_set = set(self.triples)
        # 记录原始三元组的数量
        self.num_original_triples = len(self.triples)
        # 初始数据的entity总数，relation总数
        self.num_entity = self.num_ent()
        self.num_relation = self.num_rel()
        print('entity&relation: ', self.num_entity, self.num_relation)
        # 生成异常数据
        self.bp_triples_label = self.inject_anomaly(args)
        # 正常数据+异常数据，混合，含标签信息
        self.num_triples_with_anomalies = len(self.bp_triples_label)
        # 用于生成更多的异常数据
        self.train_data, self.labels = self.get_data()
        # 等同于self.bp_triples_label
        self.triples_with_anomalies, self.triples_with_anomalies_labels = self.get_data_test()

    # def train_triples(self):
    #     return self.triples["train"]
    #
    # def valid_triples(self):
    #     return self.triples["valid"]
    #
    # def test_triples(self):
    #     return self.triples["test"]

    # def all_triples(self):
    #     return self.triples["train"] + self.triples["valid"] + self.triples["test"]

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def get_add_ent_id(self, ent):
        if ent in self.ent2id:
            ent_id = self.ent2id[ent]
        else:
            ent_id = len(self.ent2id)
            self.ent2id[ent] = ent_id
            self.id2ent[ent_id] = ent

        return ent_id

    def get_add_rel_id(self, rel):
        if rel in self.rel2id:
            rel_id = self.rel2id[rel]
        else:
            rel_id = len(self.rel2id)
            self.rel2id[rel] = rel_id
            self.id2rel[rel_id] = rel
        return rel_id

    def init_embeddings(self, entity_file, relation_file):
        entity_emb, relation_emb = [], []

        with open(entity_file) as f:
            for line in f:
                entity_emb.append([float(val) for val in line.strip().split()])

        with open(relation_file) as f:
            for line in f:
                relation_emb.append([float(val) for val in line.strip().split()])

        return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)

    def read_triples(self):
        print('Read begin!')
        for file in ["train", "valid", "test"]:
            with open(self.path + '/' + file + ".txt", "r") as f:
                for line in f.readlines():
                    try:
                        head, rel, tail = line.strip().split("\t")
                    except:
                        print(line)
                    head_id = self.get_add_ent_id(head)
                    rel_id = self.get_add_rel_id(rel)
                    tail_id = self.get_add_ent_id(tail)

                    self.triples.append((head_id, rel_id, tail_id))

                    self.A[(head_id, tail_id)] = rel_id
                    # self.A[head_id][tail_id] = rel_id

                    # generate h2t
                    if not head_id in self.h2t.keys():
                        self.h2t[head_id] = set()
                    temp = self.h2t[head_id]
                    temp.add(tail_id)
                    self.h2t[head_id] = temp

                    # generate t2h
                    if not tail_id in self.t2h.keys():
                        self.t2h[tail_id] = set()
                    temp = self.t2h[tail_id]
                    temp.add(head_id)
                    self.t2h[tail_id] = temp

        print("Read end!")
        return self.triples

    def read_triples_yago3(self):
        print('Read begin!')
        for file in ["train", "valid", "test"]:
            with open(self.path + '/' + file + ".txt", "r", encoding="utf-8") as f:
                train = f.readlines()
                # train_ = set({})
                for i in range(len(train)):
                    x = train[i].split()
                    x_ = tuple(x)
                    head, rel, tail = x_[0], x_[1], x_[2]

                    head_id = self.get_add_ent_id(head)
                    rel_id = self.get_add_rel_id(rel)
                    tail_id = self.get_add_ent_id(tail)

                    self.triples.append((head_id, rel_id, tail_id))
                    # (head_id, tail_id) 在字典中只有一个唯一对应的关系 rel_id
                    self.A[(head_id, tail_id)] = rel_id
                    # self.A[head_id][tail_id] = rel_id

                    # generate h2t
                    if not head_id in self.h2t.keys():
                        self.h2t[head_id] = set()
                    temp = self.h2t[head_id]
                    temp.add(tail_id)
                    self.h2t[head_id] = temp

                    # generate t2h
                    if not tail_id in self.t2h.keys():
                        self.t2h[tail_id] = set()
                    temp = self.t2h[tail_id]
                    temp.add(head_id)
                    self.t2h[tail_id] = temp

                del (train)

        print("Read end!")
        return self.triples

    def rand_ent_except(self, ent):
        rand_ent = random.randint(1, self.num_ent() - 1)
        while rand_ent == ent:
            rand_ent = random.randint(1, self.num_ent() - 1)
        return rand_ent

    def generate_neg_triples(self, pos_triples):
        neg_triples = []
        for head, rel, tail in pos_triples:
            head_or_tail = random.randint(0, 1)
            if head_or_tail == 0:
                new_head = self.rand_ent_except(head)
                neg_triples.append((new_head, rel, tail))
            else:
                new_tail = self.rand_ent_except(tail)
                neg_triples.append((head, rel, new_tail))
        return neg_triples

    def generate_anomalous_triples(self, pos_triples):
        neg_triples = []
        for head, rel, tail in pos_triples:
            head_or_tail = random.randint(0, 2)
            if head_or_tail == 0:
                new_head = random.randint(0, self.num_entity - 1)
                new_relation = rel
                new_tail = tail
                # neg_triples.append((new_head, rel, tail))
            elif head_or_tail == 1:
                new_head = head
                new_relation = random.randint(0, self.num_relation - 1)
                new_tail = tail
            else:
                # new_tail = self.rand_ent_except(tail)
                # neg_triples.append((head, rel, new_tail))
                new_head = head
                new_relation = rel
                new_tail = random.randint(0, self.num_entity - 1)
            anomaly = (new_head, new_relation, new_tail)
            while anomaly in self.triple_ori_set:
                if head_or_tail == 0:
                    new_head = random.randint(0, self.num_entity - 1)
                    new_relation = rel
                    new_tail = tail
                    # neg_triples.append((new_head, rel, tail))
                elif head_or_tail == 1:
                    new_head = head
                    new_relation = random.randint(0, self.num_relation - 1)
                    new_tail = tail
                else:
                    # new_tail = self.rand_ent_except(tail)
                    # neg_triples.append((head, rel, new_tail))
                    new_head = head
                    new_relation = rel
                    new_tail = random.randint(0, self.num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)
            neg_triples.append(anomaly)
        return neg_triples

    def generate_anomalous_triples_2(self, num_anomaly):
        neg_triples = []
        for i in range(num_anomaly):
            new_head = random.randint(0, self.num_entity - 1)
            new_relation = random.randint(0, self.num_relation - 1)
            new_tail = random.randint(0, self.num_entity - 1)

            anomaly = (new_head, new_relation, new_tail)

            while anomaly in self.triple_ori_set:
                new_head = random.randint(0, self.num_entity - 1)
                new_relation = random.randint(0, self.num_relation - 1)
                new_tail = random.randint(0, self.num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)

            neg_triples.append(anomaly)
        return neg_triples

    def shred_triples(self, triples):
        h_dix = [triples[i][0] for i in range(len(triples))]
        r_idx = [triples[i][1] for i in range(len(triples))]
        t_idx = [triples[i][2] for i in range(len(triples))]
        return h_dix, r_idx, t_idx

    def shred_triples_and_labels(self, triples_and_labels):
        heads = [triples_and_labels[i][0][0] for i in range(len(triples_and_labels))]
        rels = [triples_and_labels[i][0][1] for i in range(len(triples_and_labels))]
        tails = [triples_and_labels[i][0][2] for i in range(len(triples_and_labels))]
        labels = [triples_and_labels[i][1] for i in range(len(triples_and_labels))]
        return heads, rels, tails, labels

    # def all_triplets(self):
    #     ph_all, pr_all, pt_all = self.shred_triples(self.triples)
    #     nh_all, nr_all, nt_all = self.shred_triples(self.generate_neg_triples(self.triples))
    #     return ph_all, pt_all, nh_all, nt_all, pr_all

    def get_data(self):
        # bp_triples_label = self.inject_anomaly()
        bp_triples_label = self.bp_triples_label
        labels = [bp_triples_label[i][1] for i in range(len(bp_triples_label))]
        bp_triples = [bp_triples_label[i][0] for i in range(len(bp_triples_label))]
        bn_triples = self.generate_anomalous_triples(bp_triples)
        # 前一半是正常数据，后一半是异常数据
        all_triples = bp_triples + bn_triples

        return self.toarray(all_triples), self.toarray(labels)

    def get_data_test(self):
        bp_triples_label = self.bp_triples_label
        labels = [bp_triples_label[i][1] for i in range(len(bp_triples_label))]
        bp_triples = [bp_triples_label[i][0] for i in range(len(bp_triples_label))]

        return self.toarray(bp_triples), self.toarray(labels)

    def toarray(self, x):
        return torch.from_numpy(np.array(list(x)).astype(np.int32))

    def inject_anomaly(self, args):
        print("Inject anomalies!")
        original_triples = self.triples
        triple_size = len(original_triples)

        # 计算注入的异常数量
        self.num_anomalies = int(args.anomaly_ratio * self.num_original_triples)
        args.num_anomaly_num = self.num_anomalies
        print("###########Inject TOP@K% Anomalies##########")
        # if self.isInjectTopK:
        #     self.num_anomalies = args.num_anomaly_num
        #     print("###########Inject TOP@K Anomalies##########")
        # else:
        #

        # idx = random.sample(range(0, self.num_original_triples - 1), num_anomalies)
        # 随机选择一半的异常数量对应的索引，用于从原始三元组中选择三元组来生成第一部分的异常数据
        idx = random.sample(range(0, self.num_original_triples - 1), self.num_anomalies // 2)
        # 根据选定的索引从原始数据集中抽取三元组
        selected_triples = [original_triples[idx[i]] for i in range(len(idx))]
        # 生成anomalies。
        # 生成anomalies1：用已有的entities对数据替换
        # 生成anomalies2：从整个实体和关系空间中随机生成另一半的异常三元组，确保这些异常三元组不在原始数据集中
        # anomalies1和anomalies2的数据之间，它们之间可能有重复的
        anomalies = self.generate_anomalous_triples(selected_triples) \
                    + self.generate_anomalous_triples_2(self.num_anomalies // 2)

        triple_label = [(original_triples[i], 0) for i in range(len(original_triples))]
        anomaly_label = [(anomalies[i], 1) for i in range(len(anomalies))]
        # 将带有标签的原始三元组和异常三元组合并成一个列表。
        triple_anomaly_label = triple_label + anomaly_label
        # 使用 shuffle 函数将这个列表打乱，以确保数据的随机性。
        shuffle(triple_anomaly_label)
        return triple_anomaly_label

#
# dataset = Reader(args.data_dir_FB, "train")
# xxx = dataset.inject_anomaly()
# # print(xxx[0][1])
# # print(xxx[0][0])
# #
# xxx, y, a = dataset.get_data()
# print(xxx[0])
