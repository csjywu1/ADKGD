import time

import numpy as np
# import args
from dataset import Reader
# import utils
from create_batch import get_pair_batch_train, get_pair_batch_test, toarray, get_pair_batch_train_common, toarray_float
import torch
from model import BiLSTM_Attention
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import os
import logging
import math
from matplotlib import pyplot as plt
# import time
import argparse
import random
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(add_help=False)
    # args, _ = parser.parse_known_args()
    parser.add_argument('--model', default='CAGED', help='model name')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help='run training or evaluation')
    parser.add_argument('-ds', '--dataset', default='NELL-995-h25', help='dataset')
    args, _ = parser.parse_known_args()
    #模型保存到ckpt中
    parser.add_argument('--save_dir', default=f'./checkpoints/{args.dataset}/', help='model output directory')
    # parser.add_argument('--save_model', dest='save_model', action='store_true')
    # parser.add_argument('--load_model_path', default=f'./checkpoints/{args.dataset}')
    #日志信息
    parser.add_argument('--log_folder', default=f'./checkpoints/{args.dataset}/', help='model output directory')


    # data
    parser.add_argument('--data_path', default=f'./data/{args.dataset}/', help='path to the dataset')
    parser.add_argument('--dir_emb_ent', default="entity2vec.txt", help='pretrain entity embeddings')
    parser.add_argument('--dir_emb_rel', default="relation2vec.txt", help='pretrain entity embeddings')
    # parser.add_argument('--num_batch', default=2740, type=int, help='number of batch')
    parser.add_argument('--num_train', default=0, type=int, help='number of triples')
    #1个batch循环多少次
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')   #256
    parser.add_argument('--total_ent', default=0, type=int, help='number of entities')
    parser.add_argument('--total_rel', default=0, type=int, help='number of relations')

    # model architecture
    parser.add_argument('--BiLSTM_input_size', default=100, type=int, help='BiLSTM input size')
    parser.add_argument('--BiLSTM_hidden_size', default=100, type=int, help='BiLSTM hidden size')
    parser.add_argument('--BiLSTM_num_layers', default=2, type=int, help='BiLSTM layers')
    parser.add_argument('--BiLSTM_num_classes', default=1, type=int, help='BiLSTM class')
    #
    parser.add_argument('--num_neighbor', default=39, type=int, help='number of neighbors')
    parser.add_argument('--embedding_dim', default=100, type=int, help='embedding dim')

    # regularization
    parser.add_argument('--alpha', type=float, default=0.2, help='hyperparameter alpha')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout for EaGNN')

    # optimization
    parser.add_argument('--max_epoch', default=1, help='max epochs')
    parser.add_argument('--learning_rate', default=0.003, type=float, help='learning rate')
    parser.add_argument('--gama', default=0.5, type=float, help="margin parameter")
    parser.add_argument('--lam', default=0.1, type=float, help="trade-off parameter")
    #
    parser.add_argument('--mu', default=0.001, type=float, help="gated attention parameter")
    #
    parser.add_argument('--anomaly_ratio', default=0.15, type=float, help="anomaly ratio")
    #
    parser.add_argument('--num_anomaly_num', default=300, type=int, help="number of anomalies")
    args = parser.parse_args()

    # data_name = args.dataset
    # model_name = args.model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    #读取数据，训练的数据和测试的数据是分开生成
    dataset = Reader(args, args.data_path)
    if args.mode == 'train':
        train(args, dataset, device)
    elif args.mode == 'test':
        # raise NotImplementedError
        test(args, dataset, device)
    else:
        raise ValueError('Invalid mode')



def train(args, dataset, device):
    # Dataset parameters
    # data_name = args.dataset
    data_path = args.data_path
    model_name = args.model
    # 有更多异常数据的列表
    all_triples = dataset.train_data
    # labels = dataset.labels

    train_idx = list(range(len(all_triples) // 2))
    # 有异常数据的列表除迭代batch的大小
    num_iterations = math.ceil(dataset.num_triples_with_anomalies / args.batch_size)
    # 异常数据的数量
    total_num_anomalies = dataset.num_anomalies
    # 日志信息的配置，输出日志信息
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.log_folder, model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio)  + "_Neighbors" + str(args.num_neighbor) + "_" + "_log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    # There are 97653 Triples with 4650 anomalies in the graph.triples的总数和anomalies的总数
    logging.info('There are %d Triples with %d anomalies in the graph.' % (len(dataset.labels), total_num_anomalies))

    args.total_ent = dataset.num_entity
    args.total_rel = dataset.num_relation

    model_saved_path = model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio) + ".ckpt"
    model_saved_path = os.path.join(args.save_dir, model_saved_path)
    # model.load_state_dict(torch.load(model_saved_path))
    # Model BiLSTM_Attention
    model = BiLSTM_Attention(args, args.BiLSTM_input_size, args.BiLSTM_hidden_size, args.BiLSTM_num_layers, args.dropout,
                             args.alpha, args.mu, device).to(device)
    criterion = nn.MarginRankingLoss(args.gama)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epoch_times_file = os.path.join(args.log_folder, model_name + "_" + args.dataset + "_epoch_times.txt")

    # 训练6次
    for k in range(args.max_epoch):
        #每次将数据集拆分成382个batch，每个batch的大小为256，最后一个batch的大小为117
        epoch_start_time = time.time()  # 记录epoch开始时间
        for it in range(num_iterations):
            # start_read_time = time.time()
            #调用create_batch.py
            batch_h, batch_r, batch_t, batch_size = get_pair_batch_train_common(args, dataset, it, train_idx,
                                                                                args.batch_size,
                                                                                args.num_neighbor)
            # end_read_time = time.time()
            # print("Time used in loading data", it)

            batch_h = torch.LongTensor(batch_h).to(device)
            batch_t = torch.LongTensor(batch_t).to(device)
            batch_r = torch.LongTensor(batch_r).to(device)

            #bi-lstm学习到的是out，gat学习到的是out_att
            # out的维度是512 （正样本+负样本）, 600
            # out_att的维度是1024 （正样本*2+负样本*2）, 600
            out, out_att, output, output_att = model(batch_h, batch_r, batch_t)

            # 维度是（256, 2, 600）
            out = out.reshape(batch_size, -1, 2 * 3 * args.BiLSTM_hidden_size)
            # out_att 的维度变为 [256, 4, 600]
            # batch_size 被设置为 256。这意味着原始的 1024 个样本被重新分组，每组包含 4 个子样本。
            out_att = out_att.reshape(batch_size, -1, 2 * 3 * args.BiLSTM_hidden_size)

            # BiLSTM输出 正样本
            pos_h = out[:, 0, :]
            # 正常样本的头邻居信息和正常样本的尾邻居信息
            pos_z0 = out_att[:, 0, :]
            pos_z1 = out_att[:, 1, :]
            # BiLSTM输出  负样本
            neg_h = out[:, 1, :]
            # 异常样本的头邻居信息和异常样本的尾邻居信息
            neg_z0 = out_att[:, 2, :]
            neg_z1 = out_att[:, 3, :]

            # loss function
            pos_loss = args.lam * torch.norm(pos_z0 - pos_z1, p=2, dim=1) + \
                       torch.norm(pos_h[:, 0:2 * args.BiLSTM_hidden_size] +
                                  pos_h[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
                                  pos_h[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2,
                                  dim=1)
            pos_loss_score =  torch.norm(pos_h[:, 0:2 * args.BiLSTM_hidden_size] +
                                  pos_h[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
                                  pos_h[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2,
                                  dim=1)
            neg_loss = args.lam * torch.norm(neg_z0 - neg_z1, p=2, dim=1) + \
                       torch.norm(neg_h[:, 0:2 * args.BiLSTM_hidden_size] +
                                  neg_h[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
                                  neg_h[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2,
                                  dim=1)
            neg_loss_score = torch.norm(neg_h[:, 0:2 * args.BiLSTM_hidden_size] +
                                  neg_h[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
                                  neg_h[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2,
                                  dim=1)
            # y = -torch.ones(batch_size).to(device)
            # #这种方法强调了正样本的得分应该比负样本的得分低至少一个间隔
            # loss = criterion(pos_loss, neg_loss, y)

            # WU: 新视角的loss function
            output = output.reshape(batch_size, -1, args.BiLSTM_hidden_size)
            output_att = output_att.reshape(batch_size, -1, args.BiLSTM_hidden_size)

            # 两种角度正样本的KL散度
            output_pos_h = output[:, 0, :]
            output_pos_h_score = model.mlp(output_pos_h)
            output_pos_h_score = output_pos_h_score.squeeze(1)  # 从 [256, 1] 变为 [256]
            # 将 output_pos_h_score 转换为对数概率
            output_pos_h_log_prob = F.log_softmax(output_pos_h_score, dim=0)
            # pos_loss_score 是未归一化的分数，转换为概率
            pos_loss_prob = F.softmax(pos_loss_score, dim=0)
            # 计算KL散度
            kl_loss = F.kl_div(output_pos_h_log_prob, pos_loss_prob, reduction='batchmean')


            # 两种角度负样本的KL散度
            output = output.reshape(batch_size, -1, args.BiLSTM_hidden_size)
            output_att = output_att.reshape(batch_size, -1, args.BiLSTM_hidden_size)

            output_neg_h = output[:, 1, :]
            output_neg_h_score = model.mlp(output_neg_h)
            output_neg_h_score = output_neg_h_score.squeeze(1)  # 从 [256, 1] 变为 [256]
            # 将 output_pos_h_score 转换为对数概率
            output_neg_h_log_prob = F.log_softmax(output_neg_h_score, dim=0)
            # pos_loss_score 是未归一化的分数，转换为概率
            neg_loss_prob = F.softmax(neg_loss_score, dim=0)
            # 计算KL散度
            kl_loss1 = F.kl_div(output_neg_h_log_prob, neg_loss_prob, reduction='batchmean')

            # 新角度GAT的结果计算

            output_pos_z0 = output_att[:, 0, :]
            output_pos_z1 = output_att[:, 1, :]
            output_neg_z0 = output_att[:, 2, :]
            output_neg_z1 = output_att[:, 3, :]



            # 新角度的loss
            output_pos_loss = args.lam * torch.norm(output_pos_z0 - output_pos_z1, p=2, dim=1)
            output_neg_loss = args.lam * torch.norm(output_neg_z0 - output_neg_z1, p=2, dim=1)
            #两种角度之间对比的loss
            # 需要MLP降维再对比
            output_pos_z00 = model.mlp3(output_pos_z0)
            output_pos_z10 = model.mlp3(output_pos_z1)
            output_neg_z01 = model.mlp3(output_neg_z0)
            output_neg_z11 = model.mlp3(output_neg_z1)

            pos_z00 = model.mlp2(pos_z0)
            pos_z10 = model.mlp2(pos_z1)
            neg_z01 = model.mlp2(neg_z0)
            neg_z11 = model.mlp2(neg_z1)


            similar_loss1 = args.lam * torch.norm(output_pos_z00 - pos_z00, p=2, dim=1)
            similar_loss2 = args.lam * torch.norm(output_pos_z10 - pos_z10, p=2, dim=1)
            similar_loss3 = args.lam * torch.norm(output_neg_z01 - neg_z01, p=2, dim=1)
            similar_loss4 = args.lam * torch.norm(output_neg_z11 - neg_z11, p=2, dim=1)
            similar_pos_loss = similar_loss1 + similar_loss2
            similar_neg_loss = similar_loss3 + similar_loss4

            total_pos_loss = pos_loss + output_pos_loss
            total_neg_loss = neg_loss + output_neg_loss
            y = -torch.ones(batch_size).to(device)
            # 这种方法强调了正样本的得分应该比负样本的得分低至少一个间隔
            loss = criterion(total_pos_loss, total_neg_loss, y)
            # 计算平均值
            average_similar_pos_loss = torch.mean(similar_pos_loss)
            average_similar_neg_loss = torch.mean(similar_neg_loss)
            # 前面的loss主要是正负样本的之间的差异，包含了pos_loss和neg_loss (原来的角度)，以及output_pos_loss和output_neg_loss (新的角度)
            # 后面部分主要是两种角度学习上的相似。kl_loss和kl_loss1是底层，以及output_pos_loss和output_neg_loss是抽象层
            totalloss =  loss + kl_loss + kl_loss1+ average_similar_pos_loss + average_similar_neg_loss
            optimizer.zero_grad()
            totalloss.backward()
            optimizer.step()
            total_pos_loss_value = torch.sum(total_pos_loss) / (batch_size * 2.0)
            total_neg_loss_value = torch.sum(total_neg_loss) / (batch_size * 2.0)
            logging.info('There are %d Triples in this batch.' % batch_size)
            logging.info('Epoch: %d-%d, pos_loss: %f, neg_loss: %f, Loss: %f' % (
                k, it + 1, total_pos_loss_value.item(), total_neg_loss_value.item(), totalloss.item()))

            # final_time = time.time()
            # print("BP time:", math.fabs(final_time - running_time))

            torch.save(model.state_dict(), model_saved_path)

        epoch_end_time = time.time()  # 记录epoch结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # 计算epoch耗时
        logging.info('Epoch: %d, Duration: %f seconds' % (k, epoch_duration))  # 输出epoch耗时


        with open(epoch_times_file, 'a') as f:  # 将epoch耗时保存到文件
            f.write('Epoch: %d, Duration: %f seconds\n' % (k, epoch_duration))
    print("The training ends!")
    # # #
    # dataset = Reader(data_path, "test")



def test(args, dataset, device):
    # Dataset parameters
    # data_name = args.dataset
    device = torch.device('cpu')
    data_path = args.data_path
    model_name = args.model
    all_triples = dataset.train_data
    # labels = dataset.labels
    # train_idx = list(range(len(all_triples) // 2))
    num_iterations = math.ceil(dataset.num_triples_with_anomalies / args.batch_size)
    total_num_anomalies = dataset.num_anomalies
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.log_folder, model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio) + "_Neighbors" + str(args.num_neighbor) + "_" + "_log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logging.info('There are %d Triples with %d anomalies in the graph.' % (len(dataset.labels), total_num_anomalies))

    args.total_ent = dataset.num_entity
    args.total_rel = dataset.num_relation

    model_saved_path = model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio) + ".ckpt"
    model_saved_path = os.path.join(args.save_dir, model_saved_path)

    model1 = BiLSTM_Attention(args, args.BiLSTM_input_size, args.BiLSTM_hidden_size, args.BiLSTM_num_layers, args.dropout,
                              args.alpha, args.mu, device).to(device)
    #加载模型
    model1.load_state_dict(torch.load(model_saved_path))
    model1.eval()
    with torch.no_grad():
        all_loss = []
        all_label = []
        all_pred = []
        start_id = 0
        # epochs = int(len(dataset.bp_triples_label) / 100)
        # 2720
        for i in range(num_iterations):
            # start_read_time = time.time()
            # 预测当中需要用到labels
            batch_h, batch_r, batch_t, labels, start_id, batch_size = get_pair_batch_test(dataset, args.batch_size,
                                                                                          args.num_neighbor, start_id)
            # labels = labels.unsqueeze(1)
            # batch_size = input_triples.size(0)


            batch_h = torch.LongTensor(batch_h).to(device)
            batch_t = torch.LongTensor(batch_t).to(device)
            batch_r = torch.LongTensor(batch_r).to(device)
            labels = labels.to(device)
            #
            out, out_att, output, output_att = model1(batch_h, batch_r, batch_t)
            out_att = out_att.reshape(batch_size, 2, 2 * 3 * args.BiLSTM_hidden_size)
            #
            out_att_view0 = out_att[:, 0, :]
            out_att_view1 = out_att[:, 1, :]
            # [B, 600] [B, 600]

            #差异越大，越有问题。越小越没问题
            pos_loss = args.lam * torch.norm(out_att_view0 - out_att_view1, p=2, dim=1) + \
                   torch.norm(out[:, 0:2 * args.BiLSTM_hidden_size] +
                              out[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
                              out[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2, dim=1)

            output = output.reshape(batch_size, -1, args.BiLSTM_hidden_size)
            output_att = output_att.reshape(batch_size, -1, args.BiLSTM_hidden_size)
            output_pos_h = output[:, 0, :]
            output_pos_h_score = model1.mlp(output_pos_h)
            output_pos_h_score = output_pos_h_score.squeeze(1)
            output_pos_z0 = output_att[:, 0, :]
            output_pos_z1 = output_att[:, 1, :]
            output_pos_loss = args.lam * torch.norm(output_pos_z0 - output_pos_z1, p=2, dim=1)
            loss = pos_loss + output_pos_loss
            all_loss += loss
            all_label += labels

            # print('{}th test data'.format(i))
            logging.info('[Test] Evaluation on %d batch of Original graph' % i)
            # sum = labels.sum()
            # if sum < labels.size(0):
            #     # loss = -1 * loss
            #     AUC = roc_auc_score(labels.cpu(), loss.cpu())
            #     print('AUC on the {}th test images: {} %'.format(i, np.around(AUC)))

        total_num = len(all_label)


        # 9300
        max_top_k = total_num_anomalies * 2
        min_top_k = total_num_anomalies // 10
        # all_loss = torch.from_numpy(np.array(list(all_loss.to(torch.device("cpu"))).astype(np.float))

        all_loss = np.array(all_loss)
        all_loss = torch.from_numpy(all_loss)

        # 在 torch.topk 函数中，largest=True 参数指定了函数应该返回最大的 k 个元素
        # 因此，当您设置 largest=True 时，torch.topk 会从大到小排序，并返回所有元素中最大的 max_top_k 个元素
        # 这行代码会返回 all_loss 中最大的 max_top_k 个损失值及其索引，这些损失值是从大到小排列的
        # 预测这些都是异常值
        top_loss, top_indices = torch.topk(all_loss, max_top_k, largest=True, sorted=True)
        # 获取实际对应的标签，异常和正常
        top_labels = toarray([all_label[top_indices[iii]] for iii in range(len(top_indices))])
        #初始化列表 anomaly_discovered 用于累计发现的异常数量。这里通过逐个累加顶部标签来统计在不同top-k阈值下的累积异常发现数量。
        anomaly_discovered = []
        # 这段代码通过循环遍历 top_labels，逐个累加标签值来统计在不同的 top-k 阈值下累计发现的异常数量。
        # 如果是第一个元素（i == 0），则直接添加到 anomaly_discovered 列表中。
        # 对于其他元素，将当前元素与前一个累积值相加后添加到列表中。这样，anomaly_discovered 中的每个元素都代表了从第一个到当前位置的异常标签累加和。
        for i in range(max_top_k):
            if i == 0:
                anomaly_discovered.append(top_labels[i])
            else:
                anomaly_discovered.append(anomaly_discovered[i-1] + top_labels[i])

        # 这行代码每隔10个元素从 anomaly_discovered 中提取一个元素，并将这些元素组成一个新的 NumPy 数组 results_interval_10。
        # 这样做的目的是为了在每个10个间隔的点查看累积发现的异常数量，方便后续进行分析或绘图等操作。
        # 这里假设 max_top_k 是10的倍数，因此 max_top_k // 10 计算了可以完整提取的间隔次数。
        # 在 Python 中，// 是一个运算符，用于执行整数除法，即它将执行除法操作并向下取整到最接近的整数。这通常被称为“地板除”或“整数除法”。这意味着任何除法操作的结果都将是一个整数，任何小数部分都会被丢弃。
        # 930
        results_interval_10 = np.array([anomaly_discovered[i * 10] for i in range(max_top_k // 10)])

        logging.info('[Train] final results: %s' % str(results_interval_10))
        #生成一个从1到 max_top_k 的数组，用于计算每个K值的精确度和召回率。
        top_k = np.arange(1, max_top_k + 1)

        assert len(top_k) == len(anomaly_discovered), 'The size of result list is wrong'

        # 计算精确度和召回率数组。精确度 是在top-k中正确识别的异常数量除以k，召回率是正确识别的异常数量除以总的异常数量。
        # 维度是9300,9300
        precision_k = np.array(anomaly_discovered) / top_k
        #维度是9300,1
        recall_k = np.array(anomaly_discovered) * 1.0 / total_num_anomalies

        #从精确度和召回率数组中每10个元素提取一个数据点，用于生成间隔为10的精确度和召回率。
        # 维度是930,9300
        precision_interval_10 = [precision_k[i * 10] for i in range(max_top_k // 10)]
        logging.info('[Test] final Precision: %s' % str(precision_interval_10))
        recall_interval_10 = [recall_k[i * 10] for i in range(max_top_k // 10)]
        logging.info('[Test] final Recall: %s' % str(recall_interval_10))

        logging.info('K = %d' % args.max_epoch)
        # 定义一系列比率 ratios，这些比率用于计算基于数据集大小的top-k阈值，以评估模型在不同级别的异常检测阈值下的表现。
        # 假设比率为0.001，计算的K值为 0.001 * 97,653 ≈ 98。如果在这前98个数据点中有15个是实际异常（假设），那么：
        #
        # 精确度为 15 / 98
        # 召回率为 15 / 4,650
        ratios = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.20, 0.30, 0.45]
        for i in range(len(ratios)):
            #循环遍历每个比率，计算每个比率对应的k值（num_k），即数据集中原始三元组数量的一定比例，这个k值表示在多大范围内查找异常。
            num_k = int(ratios[i] * dataset.num_original_triples)
            #如果计算得到的k值超过了设置的阈值9300，就终止循环。这是因为没有足够的数据来进行更深入的分析。
            if num_k > len(anomaly_discovered):
                break

            #对于每个有效的k值，计算召回率和精确度。
            # anomaly_discovered: 预测，选取的k个异常数据中实际异常的数量
            # 精确度是在top-k中正确识别的异常数量除以k值。
            # 召回率是在top-k中正确识别的异常数量除以总异常数量，
            precision = anomaly_discovered[num_k - 1] * 1.0 / num_k
            recall = anomaly_discovered[num_k - 1] * 1.0 / total_num_anomalies

            #记录召回率和精确度的日志信息，显示在每个比率下的检测效果，并报告总异常数、在当前阈值下发现的异常数和对应的k值。这些日志对于理解和调整模型性能至关重要。
            logging.info(
                '[Test][%s][%s] Precision %f -- %f : %f' % (args.dataset, model_name, args.anomaly_ratio, ratios[i], precision))
            logging.info('[Test][%s][%s] Recall  %f-- %f : %f' % (args.dataset, model_name, args.anomaly_ratio, ratios[i], recall))
            logging.info('[Test][%s][%s] anomalies in total: %d -- discovered:%d -- K : %d' % (
                args.dataset, model_name, total_num_anomalies, anomaly_discovered[num_k - 1], num_k))

if __name__ == '__main__':
    main()

# def test(args, dataset, device):
#     # Dataset parameters
#     device = torch.device('cpu')
#     data_path = args.data_path
#     model_name = args.model
#     all_triples = dataset.train_data
#     num_iterations = math.ceil(dataset.num_triples_with_anomalies / args.batch_size)
#     total_num_anomalies = dataset.num_anomalies
#     logging.basicConfig(level=logging.INFO)
#     file_handler = logging.FileHandler(os.path.join(args.log_folder, model_name + "_" + args.dataset + "_" + str(
#         args.anomaly_ratio) + "_Neighbors" + str(args.num_neighbor) + "_" + "_log.txt"))
#     logger = logging.getLogger()
#     logger.addHandler(file_handler)
#     logging.info('There are %d Triples with %d anomalies in the graph.' % (len(dataset.labels), total_num_anomalies))
#
#     args.total_ent = dataset.num_entity
#     args.total_rel = dataset.num_relation
#
#     model_saved_path = model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio) + ".ckpt"
#     model_saved_path = os.path.join(args.save_dir, model_saved_path)
#
#     model1 = BiLSTM_Attention(args, args.BiLSTM_input_size, args.BiLSTM_hidden_size, args.BiLSTM_num_layers,
#                               args.dropout,
#                               args.alpha, args.mu, device).to(device)
#     model1.load_state_dict(torch.load(model_saved_path))
#     model1.eval()
#     with torch.no_grad():
#         all_loss = []
#         all_label = []
#         start_id = 0
#
#         for i in range(num_iterations):
#             batch_h, batch_r, batch_t, labels, start_id, batch_size = get_pair_batch_test(dataset, args.batch_size,
#                                                                                           args.num_neighbor, start_id)
#
#             batch_h = torch.LongTensor(batch_h).to(device)
#             batch_t = torch.LongTensor(batch_t).to(device)
#             batch_r = torch.LongTensor(batch_r).to(device)
#             labels = labels.to(device)
#
#             out, out_att, output, output_att = model1(batch_h, batch_r, batch_t)
#             out_att = out_att.reshape(batch_size, 2, 2 * 3 * args.BiLSTM_hidden_size)
#
#             out_att_view0 = out_att[:, 0, :]
#             out_att_view1 = out_att[:, 1, :]
#
#             pos_loss = args.lam * torch.norm(out_att_view0 - out_att_view1, p=2, dim=1) + \
#                        torch.norm(out[:, 0:2 * args.BiLSTM_hidden_size] +
#                                   out[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
#                                   out[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2, dim=1)
#
#             output = output.reshape(batch_size, -1, args.BiLSTM_hidden_size)
#             output_att = output_att.reshape(batch_size, -1, args.BiLSTM_hidden_size)
#             output_pos_h = output[:, 0, :]
#             output_pos_h_score = model1.mlp(output_pos_h)
#             output_pos_h_score = output_pos_h_score.squeeze(1)
#             output_pos_z0 = output_att[:, 0, :]
#             output_pos_z1 = output_att[:, 1, :]
#             output_pos_loss = args.lam * torch.norm(output_pos_z0 - output_pos_z1, p=2, dim=1)
#             loss = pos_loss + output_pos_loss
#             all_loss.extend(loss.cpu().numpy())
#             all_label.extend(labels.cpu().numpy())
#
#             logging.info('[Test] Evaluation on %d batch of Original graph' % i)
#
#         # Convert to numpy arrays for easier handling
#         all_loss = np.array(all_loss)
#         all_label = np.array(all_label)
#
#         # Select top 5000 scores
#         top_k = 5000
#         if len(all_loss) > top_k:
#             top_indices = np.argsort(-all_loss)[:top_k]
#             top_loss = all_loss[top_indices]
#             top_labels = all_label[top_indices]
#         else:
#             top_loss = all_loss
#             top_labels = all_label
#
#         # Save the top scores and labels to a file
#         np.savez('top_930_scores_labels_adkgd.npz', scores=top_loss, labels=top_labels)
#
#         # Create the plot
#         plt.figure(figsize=(10, 6))
#         plt.scatter(range(len(top_loss)), top_loss, c=top_labels, cmap='coolwarm', s=10, alpha=0.7)
#         plt.colorbar(label='Label')
#         plt.xlabel('Instance')
#         plt.ylabel('Trustworthiness Score')
#         plt.title('Top 5000 Node Score Distribution')
#         plt.savefig('top_5000_node_score_distribution.pdf', format='pdf', bbox_inches='tight')
#         plt.show()
#
#
# if __name__ == '__main__':
#     main()
