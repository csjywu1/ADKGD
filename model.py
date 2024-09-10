import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import logging

class IdentityMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IdentityMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层，维度从100到100
        self.relu = nn.ReLU()                        # ReLU激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 第二层，维度从100到100

    def forward(self, x):
        x = self.fc1(x)      # 通过第一层
        x = self.relu(x)     # 应用ReLU激活函数
        x = self.fc2(x)      # 通过第二层
        return x

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层从600降到300
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 第二层从300降到100

    def forward(self, x):
        x = self.fc1(x)  # 通过第一层
        x = self.relu(x)  # 应用ReLU激活函数
        x = self.fc2(x)  # 通过第二层
        return x
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1):  # 默认output_dim为1，输出单个分数
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)  # 第一个全连接层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(input_dim//2, output_dim)  # 第二个全连接层，输出维度为1

    def forward(self, x):
        x = self.fc1(x)  # 输入通过第一个全连接层
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)  # 第二个全连接层输出最终的分数
        x = torch.sigmoid(x)  # 应用sigmoid函数确保输出在[0, 1]之间
        return x  # 返回的是每个样本的分数
class ProcessLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProcessLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 假设x的形状是[batch_size, seq_len, feature_len], 即[256, 3, 100]
        out, (hidden, cell) = self.lstm(x)
        # 取最后一个时间步的输出。通过使用 out[:, -1, :]，你从每个序列中提取最后一个时间步的输出。这意味着从每个序列的最后时间点提取100维的输出向量。
        final_output = out[:, -1, :]
        return final_output


class GraphAttentionLayer2(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, mu=0.001, concat=False):
        super(GraphAttentionLayer2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.mu = mu

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp):
        """
        inp: input_fea [Batch_size, N, in_features]
        """
        # inp 的维度是 [1024, 40, 600]
        # h 的维度是 [1024, 40, 600]。这是因为 inp 的最后一维（特征维）和 W 的第一维进行了矩阵乘法，保留了 inp 的前两维和 W 的第二维。
        h = torch.matmul(inp, self.W)  # [batch_size, N, out_features]
        # N 是 40
        N = h.size()[1]
        # B 是 1024
        B = h.size()[0]  # B batch_size

        # a 是将中心节点（这里假定为每个批次中的第一个节点）的特征复制 N 次，以便每个节点都与中心节点的特征进行对比。
        # a 的维度是 [1024, 40, 600]，这是因为你取出了每批次的第一个节点的特征，并将这个特征复制了 N 次。
        a = h[:, 0, :].unsqueeze(1).repeat(1, N, 1)  # [batch_size, N, out_features]
        # a_input 是将原始特征 h 和复制的中心节点特征 a 沿特征维度拼接
        # a_input 的维度是 [1024, 40, 1200]，由于 h 和 a 在特征维度上拼接，特征维度翻倍。
        a_input = torch.cat((h, a), dim=2)  # [batch_size, N, 2*out_features]

        # 如果 self.a 的维度是 [1200, 1]，则 torch.matmul(a_input, self.a) 的结果维度是 [1024, 40, 1]。这里 1200 维的特征被映射到一个标量，表示注意力原始得分。
        e = self.leakyrelu(torch.matmul(a_input, self.a))
        # [batch_size, N, 1]

        # 标准化注意力系数
        # attention 的维度在每步操作中保持为 [1024, 40, 1]。
        # F.softmax(e, dim=1) 这一步是在维度 1 上应用 softmax 函数，确实意味着它是针对每个 batch 中的 N 个元素（这里是 40 个节点）执行 softmax。softmax 函数会计算 e 的指数，然后对这些指数求和，最后将每个指数除以这个求和结果。在这个过程中，每个批次中的每组40个元素的注意力得分都会被归一化，使得每组内的得分相加为 1。
        attention = F.softmax(e, dim=1)  # [batch_size, N, 1]
        attention = attention - self.mu
        attention = (attention + abs(attention)) / 2.0

        # print(attention)
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout
        # 经过 view 重塑后，attention 的维度变为 [1024, 1, 40]，以适应后续的矩阵乘法。
        attention = attention.view(B, 1, N)
        # torch.matmul(attention, h) 的结果维度是 [1024, 1, 600]，其中每个批次中的每个节点都通过加权的 attention 得到了一个 600 维的输出特征。
        # attention 矩阵中的每个元素表示一个注意力得分，它描述了在当前批次中每个节点相对于其它节点的重要性。在这个矩阵中，每个批次中只有一行和40列，行中的每个元素都是对应节点的注意力得分。
        # h 矩阵是转换后的特征表示，每行对应一个节点，每列是节点的特征向量。

        # 当执行 torch.matmul(attention, h) 时，实际上是对每个批次中的40个节点的600维特征向量进行加权平均。
        # 权重由 attention 中的注意力得分给出。这个操作结果是每个批次获得一个加权的特征表示，它综合了所有40个节点的信息，但通过注意力权重调整了每个节点特征的贡献。得分越高的节点对最终的特征表示贡献越大。
        # squeeze(1) 操作后，维度变为 [1024, 600]，移除了中间的单一维度。
        h_prime = torch.matmul(attention, h).squeeze(
            1)  # [batch_size, 1, N]*[batch_size, N, out_features] => [batch_size, 1, out_features]

        # F.elu(h_prime) 对 h_prime 应用 ELU (Exponential Linear Unit) 激活函数。ELU 是一种常用的激活函数，其特点是在正输入值上表现为线性，而在负输入值上提供小于但接近零的输出，有助于减少神经网络中的梯度消失问题。
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
class GraphAttentionLayer1(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, mu=0.001, concat=False):
        super(GraphAttentionLayer1, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features 
        self.dropout = dropout 
        self.alpha = alpha 
        self.concat = concat
        self.mu = mu

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  

        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp):
        """
        inp: input_fea [Batch_size, N, in_features]
        """
        # inp 的维度是 [1024, 40, 600]
        # h 的维度是 [1024, 40, 600]。这是因为 inp 的最后一维（特征维）和 W 的第一维进行了矩阵乘法，保留了 inp 的前两维和 W 的第二维。
        h = torch.matmul(inp, self.W)  # [batch_size, N, out_features]
        # N 是 40
        N = h.size()[1]
        # B 是 1024
        B = h.size()[0]  # B batch_size

        # a 是将中心节点（这里假定为每个批次中的第一个节点）的特征复制 N 次，以便每个节点都与中心节点的特征进行对比。
        # a 的维度是 [1024, 40, 600]，这是因为你取出了每批次的第一个节点的特征，并将这个特征复制了 N 次。
        a = h[:, 0, :].unsqueeze(1).repeat(1, N, 1)  # [batch_size, N, out_features]
        # a_input 是将原始特征 h 和复制的中心节点特征 a 沿特征维度拼接
        # a_input 的维度是 [1024, 40, 1200]，由于 h 和 a 在特征维度上拼接，特征维度翻倍。
        a_input = torch.cat((h, a), dim=2)  # [batch_size, N, 2*out_features]

        # 如果 self.a 的维度是 [1200, 1]，则 torch.matmul(a_input, self.a) 的结果维度是 [1024, 40, 1]。这里 1200 维的特征被映射到一个标量，表示注意力原始得分。
        e = self.leakyrelu(torch.matmul(a_input, self.a))
        # [batch_size, N, 1]

        # 标准化注意力系数
        # attention 的维度在每步操作中保持为 [1024, 40, 1]。
        # F.softmax(e, dim=1) 这一步是在维度 1 上应用 softmax 函数，确实意味着它是针对每个 batch 中的 N 个元素（这里是 40 个节点）执行 softmax。softmax 函数会计算 e 的指数，然后对这些指数求和，最后将每个指数除以这个求和结果。在这个过程中，每个批次中的每组40个元素的注意力得分都会被归一化，使得每组内的得分相加为 1。
        attention = F.softmax(e, dim=1)  # [batch_size, N, 1]
        attention = attention - self.mu
        attention = (attention + abs(attention)) / 2.0

        # print(attention)
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout
        # 经过 view 重塑后，attention 的维度变为 [1024, 1, 40]，以适应后续的矩阵乘法。
        attention = attention.view(B, 1, N)
        # torch.matmul(attention, h) 的结果维度是 [1024, 1, 600]，其中每个批次中的每个节点都通过加权的 attention 得到了一个 600 维的输出特征。
        # attention 矩阵中的每个元素表示一个注意力得分，它描述了在当前批次中每个节点相对于其它节点的重要性。在这个矩阵中，每个批次中只有一行和40列，行中的每个元素都是对应节点的注意力得分。
        # h 矩阵是转换后的特征表示，每行对应一个节点，每列是节点的特征向量。

        #当执行 torch.matmul(attention, h) 时，实际上是对每个批次中的40个节点的600维特征向量进行加权平均。
        # 权重由 attention 中的注意力得分给出。这个操作结果是每个批次获得一个加权的特征表示，它综合了所有40个节点的信息，但通过注意力权重调整了每个节点特征的贡献。得分越高的节点对最终的特征表示贡献越大。
        # squeeze(1) 操作后，维度变为 [1024, 600]，移除了中间的单一维度。
        h_prime = torch.matmul(attention, h).squeeze(1)  # [batch_size, 1, N]*[batch_size, N, out_features] => [batch_size, 1, out_features]

       # F.elu(h_prime) 对 h_prime 应用 ELU (Exponential Linear Unit) 激活函数。ELU 是一种常用的激活函数，其特点是在正输入值上表现为线性，而在负输入值上提供小于但接近零的输出，有助于减少神经网络中的梯度消失问题。
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class BiLSTM_Attention(torch.nn.Module):
    def __init__(self, args, input_size, hidden_size, num_layers, dropout, alpha, mu, device):
        super(BiLSTM_Attention, self).__init__()
        # self.ent_embeddings = nn.Embedding(args.total_ent + 1, args.embedding_dim)
        # self.rel_embeddings = nn.Embedding(args.total_rel + 1, args.embedding_dim)
        # self.init_weights()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = 3
        self.BiLSTM_input_size = args.BiLSTM_input_size
        self.num_neighbor = args.num_neighbor
        #lstm层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        #self.fc = nn.Linear(hidden_size * 2 * self.seq_length, num_classes)  # 2 for bidirection
        self.device = device
        #全连接层
        self.attention = GraphAttentionLayer1(self.hidden_size * 2 * self.seq_length, self.hidden_size * 2 * self.seq_length, dropout=dropout, alpha=alpha, mu=mu, concat=False)
        # self.attentions = [GraphAttentionLayer(self.hidden_size * 2 * self.seq_length, self.hidden_size * 2 * self.seq_length, dropout=dropout, alpha=alpha, concat=False) for _ in
        #                    range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        self.ent_embeddings = nn.Embedding(args.total_ent, args.embedding_dim)
        self.rel_embeddings = nn.Embedding(args.total_rel, args.embedding_dim)

        # print(toarray_float(ent_vec).shape)
        # print(args.total_ent, args.total_rel, args.embedding_dim)
        # self.ent_embeddings.weight.data.copy_(torch.from_numpy(ent_vec))
        # self.rel_embeddings.weight.data.copy_(torch.from_numpy(rel_vec))
        uniform_range = 6 / np.sqrt(args.embedding_dim)
        self.ent_embeddings.weight.data.uniform_(-uniform_range, uniform_range)
        self.rel_embeddings.weight.data.uniform_(-uniform_range, uniform_range)

        #WU: 用于生成另一个视角的embeddings
        self.ent_embeddings1 = nn.Embedding(args.total_ent, args.embedding_dim)
        self.rel_embeddings1 = nn.Embedding(args.total_rel, args.embedding_dim)
        uniform_range = 6 / np.sqrt(args.embedding_dim)
        self.ent_embeddings1.weight.data.uniform_(-uniform_range, uniform_range)
        self.rel_embeddings1.weight.data.uniform_(-uniform_range, uniform_range)

        self.process_lstm = ProcessLSTM(args.embedding_dim, args.embedding_dim, args.embedding_dim)

        self.hidden_size1 = 100

        self.attention2 = GraphAttentionLayer2(self.hidden_size1, self.hidden_size1, dropout=dropout, alpha=alpha, mu=mu, concat=False)

        self.mlp = MLP(self.hidden_size1)
        self.mlp2 = TwoLayerMLP(self.hidden_size * 2 * self.seq_length, self.hidden_size * 2 * self.seq_length // 2, self.hidden_size1)
        self.mlp3 = IdentityMLP(self.hidden_size1,self.hidden_size1,self.hidden_size1)


    def forward(self, batch_h, batch_r, batch_t):
        #原来角度的学习
        head = self.ent_embeddings(batch_h)
        relation = self.rel_embeddings(batch_r)
        tail = self.ent_embeddings(batch_t)

        batch_triples_emb = torch.cat((head, relation), dim=1)
        batch_triples_emb = torch.cat((batch_triples_emb, tail), dim=1)
        x = batch_triples_emb.view(-1, 3, self.BiLSTM_input_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)# 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (B, seq_length, hidden_size*2)
        out = out.reshape(-1, self.hidden_size * 2 * self.seq_length)
        out = out.reshape(-1, self.num_neighbor + 1, self.hidden_size * 2 * self.seq_length)
        out_att = self.attention(out)
        out = out.reshape(-1, self.num_neighbor * 2 + 2, self.hidden_size * 2 * self.seq_length)

        # WU: 另一个角度的学习
        head1 = self.ent_embeddings1(batch_h)
        relation1 = self.rel_embeddings1(batch_r)
        tail1 = self.ent_embeddings1(batch_t)

        batch_triples_emb1 = torch.cat((head1, relation1), dim=1)
        batch_triples_emb1 = torch.cat((batch_triples_emb1, tail1), dim=1)
        x1 = batch_triples_emb1.view(-1, 3, self.BiLSTM_input_size)
        output = self.process_lstm(x1) #输入维度（40960,3,100），输出维度(40960, 100)
        # attention部分
        # output = output.reshape(-1, self.hidden_size * 2)
        output = output.reshape(-1, self.num_neighbor + 1, self.hidden_size1)
        output_att = self.attention2(output)
        output = output.reshape(-1, self.num_neighbor * 2 + 2, self.hidden_size1)


        # 返回了正样本和负样本 #out[512,80,100],out_att[1024,600], output[512,80,100],output_att[1024,100]
        return out[:, 0, :], out_att, output[:, 0, :], output_att
