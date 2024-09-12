import math
import torch
import collections
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch.autograd import Variable


def clone_module_to_modulelist(module, module_num):
    """
    克隆n个Module类放入ModuleList中，并返回ModuleList，这个ModuleList中的每个Module都是一模一样的
    nn.ModuleList，它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。
    你可以把任意 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面，
    加入到 nn.ModuleList 里面的 module 是会自动注册到整个网络上的，
    同时 module 的 parameters 也会自动添加到整个网络中。
    :param module: 被克隆的module
    :param module_num: 被克隆的module数
    :return: 装有module_num个相同module的ModuleList
    """
    return nn.ModuleList([deepcopy(module) for _ in range(module_num)])


##----------------------------------Embedding模块--------------------------------------##
class CNN_Embedding(nn.Module):
    def __init__(self):
        super(CNN_Embedding, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 22), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        # self.layer2_dim = 250
        self.padding1 = nn.ZeroPad2d((10, 9, 0, 1))  # 左边添加10，右边添加9,时间维扩充19
        # self.conv2 = nn.Conv2d(1, 4, (2, 20),stride=(2,2))   # 输出(2,4,8,250）
        # test1
        self.conv2 = nn.Conv2d(1, 4, (2, 20), stride=(4, 2))  # 输出(2,4,8,250）
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d((1, 5))  # (2,4,8,250）

        # Layer 3
        # self.conv3 = nn.Conv2d(4, 1, (2, 2),stride=(2,2))
        self.conv3 = nn.Conv2d(4, 8, (4, 1), stride=(2, 1))
        self.pooling3 = nn.MaxPool2d((1, 2))

    def forward(self, x):
        x = x.unsqueeze(1)  # （batch,1,22,1000)
        x = x.permute(0, 1, 3, 2)  # (batch,1,1000,22)
        # Layer 1
        x = F.elu(self.conv1(x))  # (batch,16,1000,1)
        x = self.batchnorm1(x)  # (batch,16,1000,1)
        x = F.dropout(x, 0.3)
        x = x.permute(0, 3, 1, 2)  # (batch,1,16,1000)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.3)
        x = self.pooling2(x)  # (batch,4,8,100)

        # Layer 3
        # x = self.padding2(x)
        x = F.elu(self.conv3(x))
        # x = F.dropout(x, 0.3)
        x = self.pooling3(x)

        # x = torch.squeeze(x, dim=1)    # 经过Layer 3 输出的维度为（batch,5,50)
        x = torch.squeeze(x, dim=2)
        return x


class CNN_Embedding_Deepconv(nn.Module):
    def __init__(self, in_channels, out_channels=30, dropout=0.2):
        super(CNN_Embedding_Deepconv, self).__init__()
        self.dropout = dropout
        # Layer 1
        self.conv1 = nn.Conv2d(1, out_channels, (1, 13), padding=0)

        # Layer 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, (in_channels, 1), bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 30), stride=(1, 4))

    def forward(self, x):
        x = x.unsqueeze(1)
        # Layer 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        x = x.squeeze(2)
        # x = x.reshape(-1, x.shape[1] * x.shape[2])
        return x


class CNN_Embedding_Deepconv_2b(nn.Module):
    def __init__(self, in_channels, out_channels=30, dropout=0.2):
        super(CNN_Embedding_Deepconv_2b, self).__init__()
        self.dropout = dropout
        # Layer 1
        self.conv1 = nn.Conv2d(1, out_channels, (1, 22), padding=0)

        # Layer 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, (in_channels, 1), bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 30), stride=(1, 4))

    def forward(self, x):
        x = x.unsqueeze(1)
        # Layer 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        x = x.squeeze(2)
        # x = x.reshape(-1, x.shape[1] * x.shape[2])
        return x


##----------------------------------Transformer模块------------------------------------##

class LayerNorm(nn.Module):

    def __init__(self, feature, eps=1e-6):
        """
        :param feature: self-attention 的 x 的大小
        :param eps:
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    这不仅仅做了残差，这是把残差和 layernorm 一起给做了

    """

    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        # 第一步做 layernorm
        self.layer_norm = LayerNorm(size)
        # 第二步做 dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        :param x: 就是self-attention的输入
        :param sublayer: self-attention层
        :return:
        """
        return self.dropout(self.layer_norm(x + sublayer(x)))


def self_attention(query, key, value, dropout=None, mask=None):
    """
    自注意力计算
    :param query: Q
    :param key: K
    :param value: V
    :param dropout: drop比率
    :param mask: 是否mask
    :return: 经自注意力机制计算后的值
    """
    d_k = query.size(-1)  # 防止softmax未来求梯度消失时的d_k
    # Q,K相似度计算公式：\frac{Q^TK}{\sqrt{d_k}}
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算
    # 判断是否要mask，注：mask的操作在QK之后，softmax之前
    if mask is not None:
        """
        scores.masked_fill默认是按照传入的mask中为1的元素所在的索引，
        在scores中相同的的索引处替换为value，替换值为-1e9，即-(10^9)
        """
        # mask.cuda()
        # 进行mask操作，由于参数mask==0，因此替换上述mask中为0的元素所在的索引

        scores = scores.masked_fill(mask == 0, -1e9)

    self_attn_softmax = F.softmax(scores, dim=-1)  # 进行softmax
    # 判断是否要对相似概率分布进行dropout操作
    if dropout is not None:
        self_attn_softmax = dropout(self_attn_softmax)

    # 注意：返回经自注意力计算后的值，以及进行softmax后的相似度（即相似概率分布）
    return torch.matmul(self_attn_softmax, value), self_attn_softmax


def self_attention(query, key, value, mode, dropout=None, mask=None):
    """
    自注意力计算
    :param query: Q
    :param key: K
    :param value: V
    :param dropout: drop比率
    :param mask: 是否mask
    :param mode: 模式切换，分为channel模式和time时序模式,"C"为channel模式，“T”为时序模式
    :return: 经自注意力机制计算后的值
    """
    if mode == "C":
        d_k = query.size(-1)  # 防止softmax未来求梯度消失时的d_k
        # Q,K相似度计算公式：\frac{Q^TK}{\sqrt{d_k}}
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算
    else:
        d_k = query.size(-2)  # 防止softmax未来求梯度消失时的d_k
        # Q,K相似度计算公式：\frac{Q^TK}{\sqrt{d_k}}
        scores = torch.matmul(query.transpose(-2, -1), key) / math.sqrt(d_k)  # Q,K相似度计算
    # 判断是否要mask，注：mask的操作在QK之后，softmax之前
    if mask is not None:
        """
        scores.masked_fill默认是按照传入的mask中为1的元素所在的索引，
        在scores中相同的的索引处替换为value，替换值为-1e9，即-(10^9)
        """
        # mask.cuda()
        # 进行mask操作，由于参数mask==0，因此替换上述mask中为0的元素所在的索引

        scores = scores.masked_fill(mask == 0, -1e9)

    self_attn_softmax = F.softmax(scores, dim=-1)  # 进行softmax
    # 判断是否要对相似概率分布进行dropout操作
    if dropout is not None:
        self_attn_softmax = dropout(self_attn_softmax)

    if mode == "C":
        attention_output = torch.matmul(self_attn_softmax, value)
    elif mode == "T":
        attention_output = torch.matmul(self_attn_softmax, value.transpose(-2, -1)).transpose(-2, -1)
    # 注意：返回经自注意力计算后的值，以及进行softmax后的相似度（即相似概率分布）
    return attention_output, self_attn_softmax


class MultiHeadAttention(nn.Module):
    """
    多头注意力计算
    """

    def __init__(self, head, d_model, mode="C", dropout=0.1):
        """
        :param head: 头数
        :param d_model: 词向量的维度，必须是head的整数倍
        :param dropout: drop比率
        """
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)  # 确保词向量维度是头数的整数倍
        self.d_k = d_model // head  # 被拆分为多头后的某一头词向量的维度
        self.head = head
        self.d_model = d_model
        self.mode = mode

        """
        由于多头注意力机制是针对多组Q、K、V，因此有了下面这四行代码，具体作用是，
        针对未来每一次输入的Q、K、V，都给予参数进行构建
        其中linear_out是针对多头汇总时给予的参数
        """
        self.linear_query = nn.Linear(d_model, d_model)  # 进行一个普通的全连接层变化，但不修改维度
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_softmax = None  # attn_softmax是能量分数, 即句子中某一个词与所有词的相关性分数， softmax(QK^T)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            """
            多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
            再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第二维（head维）添加一维，与后面的self_attention计算维度一样
            具体点将，就是：
            因为mask的作用是未来传入self_attention这个函数的时候，作为masked_fill需要mask哪些信息的依据
            针对多head的数据，Q、K、V的形状维度中，只有head是通过view计算出来的，是多余的，为了保证mask和
            view变换之后的Q、K、V的形状一直，mask就得在head这个维度添加一个维度出来，进而做到对正确信息的mask
            """
            mask = mask.unsqueeze(1)

        n_batch = query.size(0)  # batch_size大小，假设query的维度是：[10, 32, 512]，其中10是batch_size的大小

        """
        下列三行代码都在做类似的事情，对Q、K、V三个矩阵做处理
        其中view函数是对Linear层的输出做一个形状的重构，其中-1是自适应（自主计算）
        从这种重构中，可以看出，虽然增加了头数，但是数据的总维度是没有变化的，也就是说多头是对数据内部进行了一次拆分
        transopose(1,2)是对前形状的两个维度(索引从0开始)做一个交换，例如(2,3,4,5)会变成(2,4,3,5)
        因此通过transpose可以让view的第二维度参数变成n_head
        假设Linear成的输出维度是：[10, 32, 512]，其中10是batch_size的大小
        注：这里解释了为什么d_model // head == d_k，如若不是，则view函数做形状重构的时候会出现异常
        """
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]，head=8
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]

        # x是通过自注意力机制计算出来的值， self.attn_softmax是相似概率分布
        x, self.attn_softmax = self_attention(query, key, value, self.mode, dropout=self.dropout, mask=mask)

        """
        下面的代码是汇总各个头的信息，拼接后形成一个新的x
        其中self.head * self.d_k，可以看出x的形状是按照head数拼接成了一个大矩阵，然后输入到linear_out层添加参数
        contiguous()是重新开辟一块内存后存储x，然后才可以使用.view方法，否则直接使用.view方法会报错
        """
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        return self.linear_out(x)


class FeedForward(nn.Module):
    """
    两层具有残差网络的前馈神经网络，FNN网络
    """

    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        """
        :param d_model: FFN第一层输入的维度
        :param d_ff: FNN第二层隐藏层输入的维度
        :param dropout: drop比率
        """
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: 输入数据，形状为(batch_size, input_len, model_dim)
        :return: 输出数据（FloatTensor），形状为(batch_size, input_len, model_dim)
        """
        inter = self.dropout_1(self.elu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        # return output + x   # 即为残差网络
        return output  # + x


class SublayerConnection(nn.Module):
    """
    子层的连接: layer_norm(x + sublayer(x))
    上述可以理解为一个残差网络加上一个LayerNorm归一化
    """

    def __init__(self, size, dropout=0.1):
        """
        :param size: d_model
        :param dropout: drop比率
        """
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        # TODO：在SublayerConnection中LayerNorm可以换成nn.BatchNorm2d
        # self.layer_norm = nn.BatchNorm2d()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.layer_norm(x + sublayer(x)))


class EncoderLayer(nn.Module):
    """
    一层编码Encoder层
    MultiHeadAttention -> Add & Norm -> Feed Forward -> Add & Norm
    """

    def __init__(self, size, attn, feed_forward, dropout=0.1):
        """
        :param size: d_model
        :param attn: 已经初始化的Multi-Head Attention层
        :param mode: Attention模块的计算模式
        :param feed_forward: 已经初始化的Feed Forward层
        :param dropout: drop比率
        """
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        """
        下面一行的作用是因为一个Encoder层具有两个残差结构的网络
        因此构建一个ModuleList存储两个SublayerConnection，以便未来对数据进行残差处理
        """
        self.sublayer_connection_list = clone_module_to_modulelist(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        """
        :param x: Encoder层的输入
        :param mask: mask标志
        :return: 经过一个Encoder层处理后的输出
        """
        """
        编码层第一层子层
        self.attn 应该是一个已经初始化的Multi-Head Attention层
        把Encoder的输入数据x和经过一个Multi-Head Attention处理后的x_attn送入第一个残差网络进行处理得到first_x
        """
        first_x = self.sublayer_connection_list[0](x, lambda x_attn: self.attn(x, x, x, mask))

        """
        编码层第二层子层
        把经过第一层子层处理后的数据first_x与前馈神经网络送入第二个残差网络进行处理得到Encoder层的输出
        """
        return self.sublayer_connection_list[1](first_x, self.feed_forward)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=22):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


class Encoder(nn.Module):
    """
    构建n层编码层
    """

    def __init__(self, n, encoder_layer):
        """
        :param n: Encoder层的层数
        :param encoder_layer: 初始化的Encoder层
        """
        super(Encoder, self).__init__()
        self.encoder_layer_list = clone_module_to_modulelist(encoder_layer, n)

    def forward(self, x, src_mask):
        """
        :param x: 输入数据
        :param src_mask: mask标志
        :return: 经过n层Encoder处理后的数据
        """
        for encoder_layer in self.encoder_layer_list:
            x = encoder_layer(x, src_mask)
        return x


class ST_Transformer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout):
        super(ST_Transformer, self).__init__()
        attn_T = MultiHeadAttention(n_heads, d_model, "T", dropout)  # 多头注意力计算
        attn_C = MultiHeadAttention(n_heads, d_model, "C", dropout)  # 多头注意力计算
        feed_forward = FeedForward(d_model, d_ff)
        # 时间维encoder
        self.encoder_T = Encoder(n_layers, EncoderLayer(d_model, deepcopy(attn_T), deepcopy(feed_forward)))
        # 空间维encoder
        self.encoder_C = Encoder(n_layers, EncoderLayer(d_model, deepcopy(attn_C), deepcopy(feed_forward)))
        self.elu = nn.ELU()
        self.softmax = nn.Softmax()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # 1, 1,
        self.fc1 = nn.Linear(d_model, 16)
        # 尝试只过一层MLP 输出cls_token
        self.fc2 = nn.Linear(16, 4)  # 四分类，原来是3分类

    def forward(self, x):
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)  # bz x 1 x 750
        x = torch.cat((x, cls_token), 1)  # bz x 23 x 750
        # x = self.encoder_T(x,None)   # 若干层时间
        x = self.encoder_C(x, None)  # 若干层空间的

        # 尝试多种方式
        # mothod1 channel 维度降维
        # mothod2 reshape 一下

        cls_token = x[:, -1, :].squeeze(1)  # bz * 750    vit
        cls_token = self.elu(self.fc1(cls_token))  # MLP    512
        cls_token = self.fc2(cls_token)
        # cls_token = self.softmax(cls_token)   # pytorch 求crossentropy不需要softmax
        return cls_token


class Embedding_ST_Trans(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout):
        super(Embedding_ST_Trans, self).__init__()
        self.embedding = CNN_Embedding()
        self.trans = ST_Transformer(d_model, d_ff, n_heads, n_layers, dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.trans(x)
        return x


## ST_Trans as feature extraction 1
class ST_Transformer_feature_extractor(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout, type='all'):
        super(ST_Transformer_feature_extractor, self).__init__()
        attn_T = MultiHeadAttention(n_heads, d_model, "T", dropout)  # 多头注意力计算
        attn_C = MultiHeadAttention(1, d_model, "C", dropout)  # 多头注意力计算,空间的就用1头就可以了
        feed_forward = FeedForward(d_model, d_ff)
        # Positional-Encoding
        self.pe = PositionalEncoding(d_model, dropout, 8)  # 8是根据输入的EEG embedding 通道数决定的
        # 时间维encoder
        self.encoder_T = Encoder(n_layers, EncoderLayer(d_model, deepcopy(attn_T), deepcopy(feed_forward)))
        # 空间维encoder
        self.encoder_C = Encoder(n_layers, EncoderLayer(d_model, deepcopy(attn_C), deepcopy(feed_forward)))
        self.elu = nn.ELU()
        self.softmax = nn.Softmax()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # 1, 1,
        self.type = type
        # self.fc1 = nn.Linear(d_model, 128)
        # 尝试只过一层MLP 输出cls_token

    def forward(self, x):
        # cls_token = self.cls_token.repeat(x.shape[0], 1, 1)  # bz x 1 x 750
        # x = torch.cat((x, cls_token), 1)  # bz x 23 x 750
        if self.type == "C":
            Attention_S = self.encoder_C(x, None)  # 若干层空间的
            feature_represent = Attention_S
        elif self.type == "T":
            x_pe = self.pe(x)
            Attention_T = self.encoder_T(x_pe, None)  # 若干层时间
            feature_represent = Attention_T
        else:
            Attention_S = self.encoder_C(x, None)  # 若干层空间的
            x_pe = self.pe(x)
            Attention_T = self.encoder_T(x_pe, None)  # 若干层时间
            feature_represent = 0.8 * Attention_T + 0.2 * Attention_S
        # cls_token = x[:, -1, :].squeeze(1)  # bz * 750    vit
        # cls_token = self.elu(self.fc1(cls_token))  # MLP    512
        # cls_token = self.softmax(cls_token)   # pytorch 求crossentropy不需要softmax
        # return cls_token
        return feature_represent


## ST_Trans as feature extractor 2
class ST_Transformer_feature_extractor_sec(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout, emb_feature, type='all'):
        super(ST_Transformer_feature_extractor_sec, self).__init__()
        attn_T = MultiHeadAttention(n_heads, d_model, "T", dropout)  # 多头注意力计算
        attn_C = MultiHeadAttention(1, d_model, "C", dropout)  # 多头注意力计算,空间的就用1头就可以了
        feed_forward = FeedForward(d_model, d_ff)
        # Positional-Encoding
        self.pe = PositionalEncoding(d_model, dropout, 8)  # 8是根据输入的EEG embedding 通道数决定的
        # 时间维encoder
        self.encoder_T = Encoder(n_layers, EncoderLayer(d_model, deepcopy(attn_T), deepcopy(feed_forward)))
        # 空间维encoder
        self.encoder_C = Encoder(n_layers, EncoderLayer(d_model, deepcopy(attn_C), deepcopy(feed_forward)))
        # 时间维通道压缩
        self.average_pooling = nn.AvgPool2d(kernel_size=(2, 4))
        self.time_squeeze = nn.AvgPool2d(kernel_size=(30, 1))
        # self.fc_c = nn.Linear()
        self.elu = nn.ELU()
        self.softmax = nn.Softmax()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # 1, 1,
        self.type = type
        # self.fc1 = nn.Linear(d_model, 128)
        # 尝试只过一层MLP 输出cls_token

    def forward(self, x):
        # cls_token = self.cls_token.repeat(x.shape[0], 1, 1)  # bz x 1 x 750
        # x = torch.cat((x, cls_token), 1)  # bz x 23 x 750
        if self.type == "C":
            Attention_S = self.encoder_C(x, None)  # 若干层空间的
            feature_represent = Attention_S
        elif self.type == "T":
            x_pe = self.pe(x)
            Attention_T = self.encoder_T(x_pe, None)  # 若干层时间
            feature_represent = Attention_T
        else:
            # Method1 堆叠   0.50
            # Attention_S = self.encoder_C(x, None)  # 若干层空间的
            # Attention_C = self.encoder_T(x, None)  # 时间
            # representation = Attention_C + Attention_S + x  # 残差
            # representation = representation.unsqueeze(1)
            # # 压缩，防止参数过大
            # representation = self.average_pooling(representation)
            # # reshape操作
            # representation = representation.reshape(-1,representation.shape[2]*representation.shape[3])

            # Method2 cls token + 时间 + 残差

            channel_att_token = self.cls_token.repeat(x.shape[0], 1, 1)  # bz x 1 x 750
            channel_x = torch.cat((x, channel_att_token), 1)  # bz x 23 x 750
            Attention_C = self.encoder_C(channel_x, None)
            channel_att_token = Attention_C[:, -1, :].unsqueeze(1)

            # 时间维度的token   降低最终计算量   240输出
            # x = x.permute(0,2,1)
            squeeze_x = self.time_squeeze(x)
            Attention_T = self.encoder_T(squeeze_x, None)  # 若干层时间
            representation = channel_att_token + Attention_T + squeeze_x
            representation = representation.squeeze(1)

            # 空间维度的token
            # channel_att_token = self.cls_token.repeat(x.shape[0], 1, 1)  # bz x 1 x 750
            # channel_x = torch.cat((x, channel_att_token), 1)  # bz x 23 x 750
            # Attention_S = self.encoder_C(x, None)  # 若干层空间的
            # channel_att_token = Attention_S[:, -1, :].unsqueeze(1)
            # feature_represent = Attention_S[:, -1, :].squeeze(1)  # bz * 750    vit
            # x_pe = self.pe(x)
            # 时间维度的token   降低最终计算量
            # x = x.permute(0,2,1)
            # temp_x = self.average_pooling(x).permute(0,2,1)
            # Attention_T = self.encoder_T(temp_x, None)   # 若干层时间
            # representation = Attention_T.squeeze(1)
            # temp_att_token = Attention_T.squeeze(1)
            # 将通道维度和时间维度转换
            # Attention_T = temp_x.permute(0,2,1)
            # temp_att_token = self.average_pooling(Attention_T)
            # temp_att_token = temp_att_token.permute(0,2,1).squeeze(1)  # 还原维度
            # feature_represent = Attention_T + channel_att_token
            # represent_token = torch.cat([channel_att_token,temp_att_token], axis = 1)
        return representation


## ST_Trans as feature extractor 2
class ST_Transformer_feature_extractor_2b(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout, emb_feature, type='all'):
        super(ST_Transformer_feature_extractor_2b, self).__init__()
        attn_T = MultiHeadAttention(n_heads, d_model, "T", dropout)  # 多头注意力计算
        attn_C = MultiHeadAttention(1, d_model, "C", dropout)  # 多头注意力计算,空间的就用1头就可以了
        feed_forward = FeedForward(d_model, d_ff)
        # Positional-Encoding
        self.pe = PositionalEncoding(d_model, dropout, 8)  # 8是根据输入的EEG embedding 通道数决定的
        # 时间维encoder
        self.encoder_T = Encoder(n_layers, EncoderLayer(d_model, deepcopy(attn_T), deepcopy(feed_forward)))
        # 空间维encoder
        self.encoder_C = Encoder(n_layers, EncoderLayer(d_model, deepcopy(attn_C), deepcopy(feed_forward)))
        # 时间维通道压缩
        self.average_pooling = nn.AvgPool2d(kernel_size=(2, 4))
        self.time_squeeze = nn.AvgPool2d(kernel_size=(30, 1))
        # self.fc_c = nn.Linear()
        self.elu = nn.ELU()
        self.softmax = nn.Softmax()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # 1, 1,
        self.type = type
        # self.fc1 = nn.Linear(d_model, 128)
        # 尝试只过一层MLP 输出cls_token

    def forward(self, x):
        # cls_token = self.cls_token.repeat(x.shape[0], 1, 1)  # bz x 1 x 750
        # x = torch.cat((x, cls_token), 1)  # bz x 23 x 750
        if self.type == "C":
            Attention_S = self.encoder_C(x, None)  # 若干层空间的
            feature_represent = Attention_S
        elif self.type == "T":
            x_pe = self.pe(x)
            Attention_T = self.encoder_T(x_pe, None)  # 若干层时间
            feature_represent = Attention_T
        else:
            # Method1 堆叠
            Attention_S = self.encoder_C(x, None)  # 若干层空间的
            Attention_C = self.encoder_T(x, None)  # 时间
            representation = Attention_C + Attention_S + x  # 残差
            representation = representation.unsqueeze(1)
            # 压缩，防止参数过大
            representation = self.average_pooling(representation)
            # reshape操作
            representation = representation.reshape(-1, representation.shape[2] * representation.shape[3])

            # Method2 cls token + 时间 + 残差

            # channel_att_token = self.cls_token.repeat(x.shape[0], 1, 1)  # bz x 1 x 750
            # channel_x = torch.cat((x, channel_att_token), 1)  # bz x 23 x 750
            # Attention_C = self.encoder_C(channel_x,None)
            # channel_att_token = Attention_C[:, -1, :].unsqueeze(1)

            # 时间维度的token   降低最终计算量   240输出
            # x = x.permute(0,2,1)
            # squeeze_x = self.time_squeeze(x)
            # Attention_T = self.encoder_T(squeeze_x, None)   # 若干层时间
            # representation = channel_att_token + Attention_T + squeeze_x
            # representation = representation.squeeze(1)

            # 空间维度的token
            # channel_att_token = self.cls_token.repeat(x.shape[0], 1, 1)  # bz x 1 x 750
            # channel_x = torch.cat((x, channel_att_token), 1)  # bz x 23 x 750
            # Attention_S = self.encoder_C(x, None)  # 若干层空间的
            # channel_att_token = Attention_S[:, -1, :].unsqueeze(1)
            # feature_represent = Attention_S[:, -1, :].squeeze(1)  # bz * 750    vit
            # x_pe = self.pe(x)
            # 时间维度的token   降低最终计算量
            # x = x.permute(0,2,1)
            # temp_x = self.average_pooling(x).permute(0,2,1)
            # Attention_T = self.encoder_T(temp_x, None)   # 若干层时间
            # representation = Attention_T.squeeze(1)
            # temp_att_token = Attention_T.squeeze(1)
            # 将通道维度和时间维度转换
            # Attention_T = temp_x.permute(0,2,1)
            # temp_att_token = self.average_pooling(Attention_T)
            # temp_att_token = temp_att_token.permute(0,2,1).squeeze(1)  # 还原维度
            # feature_represent = Attention_T + channel_att_token
            # represent_token = torch.cat([channel_att_token,temp_att_token], axis = 1)
        return representation


if __name__ == "__main__":
    # 在这个例子中模拟的输入是（batch，channel，time）
    # queries, keys, values = torch.normal(0, 1, (2, 22, 512)), torch.ones((2, 22, 512)),torch.ones((2, 22, 512))
    inp = torch.autograd.Variable(torch.randn(2, 19, 1024))
    dropout = 0.3
    n_heads = 4
    d_model = 256
    d_ff = 512
    ff_hide = 1024
    mode1 = "T"
    mode2 = "C"
    n_layers = 3
    emb_feature = 20
    model = ST_Transformer_feature_extractor_2b(d_model, d_ff, n_heads, n_layers, dropout, emb_feature, type='all')
    # model = CNN_Embedding()
    # output = model(inp)
    # pos = PositionalEncoding(d_model, 0.1)
    # res = pos(inp)
    # Attention1 = MultiHeadAttention(head,d_model,mode1)
    # Attention2 = MultiHeadAttention(head, d_model, mode2)
    # out = Attention2(inp,inp,inp)
    # FF = FeedForward(d_model,ff_hide)
    # EncoderLayer = EncoderLayer(d_model,Attention1,FF)
    # Encoder1 = Encoder(n_layer,EncoderLayer(d_model,deepcopy(Attention1),deepcopy(FF)))   ## 这里deepcopy是必要的
    # Encoder2 = Encoder(n_layer, EncoderLayer(d_model, deepcopy(Attention2), deepcopy(FF)))  ## 这里deepcopy是必要的
    # TimeCNN = CNN(22,20,10,0.1,100)

    # new module
    # model = Embedding_ST_Trans(d_model,d_ff,head,n_layer,0.1)
    # out = model(inp)
    # print(out.shape)
    # out = TimeCNN(inp)
    # out = Encoder1(inp,None)