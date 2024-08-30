import os
import sys
import logging
from torch.utils.data import Dataset, DataLoader
import torch
import dgl
from utils import *
import pickle
import torch.nn as nn
import math
import pandas as pd
import os
import sys
from torch import nn
from dgl.nn.pytorch import GATv2Conv
from dgl.nn import GlobalAttentionPooling
import math
import numpy as np
from utils import *
import time
import copy


os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dilation=3, dev="cpu"):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(len(kernel_sizes)):
            dilation_size = dilation ** i
            kernel_size = kernel_sizes[i]
            padding = (kernel_size-1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding), 
                       nn.BatchNorm1d(out_channels), nn.ReLU(), Chomp1d(padding)]
            
        self.network = nn.Sequential(*layers)
        
        self.out_dim = num_channels[-1]
        self.network.to(dev)
        
    
    def forward(self, x): #[batch_size, T, in_dim]
        x = x.permute(0,2,1) #[batch_size, out_dim, T]
        out = self.network(x)
        out = out.permute(0, 2, 1) #[batch_size, T, out_dim]

        return out


class SelfAttention(nn.Module):

    def __init__(self, input_size, seq_len):
        """
        Args:
            input_size: int, hidden_size * num_directions
            seq_len: window_size
        """
        super(SelfAttention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x):

        input_tensor = x.transpose(1, 0)  # w x b x h

        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x b x out

        input_tensor = input_tensor.transpose(1, 0)

        atten_weight = torch.nn.functional.softmax(input_tensor, dim=1)
        atten_weight = atten_weight.expand(x.shape)

        weighted_sum = x * atten_weight
        
        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, linear_sizes):
        super(FullyConnected, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i-1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor): #[batch_size, in_dim]
        return self.net(x)
    


class LatencyModel(nn.Module):
    def __init__(self, input_dim=20, trace_hiddens=[32, 64], trace_kernel_sizes=[3, 3], self_attn=True, chunk_length=30, trace_dropout=0.1, device='cpu', **kwargs):
        super(LatencyModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(input_dim, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, 
                           dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_length is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)
        

    def forward(self, x):  # [bz, T, input_dim]
        hidden_states = self.net(x)
        
        if self.self_attn: 
            hidden_states_after_self_attention = self.attn_layer(hidden_states)
            
            return hidden_states_after_self_attention
        
        return hidden_states[:, -1, :]  # [bz, out_dim]
    
class CpuModel(nn.Module):
    def __init__(self, input_dim=20, trace_hiddens=[32, 64], trace_kernel_sizes=[3, 3], self_attn=True, chunk_length=30, trace_dropout=0.1, device='cpu', **kwargs):
        super(CpuModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(input_dim, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, 
                           dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_length is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)
        

    def forward(self, x):  # [bz, T, input_dim]
        hidden_states = self.net(x)
        
        if self.self_attn: 
            hidden_states_after_self_attention = self.attn_layer(hidden_states)
            
            return hidden_states_after_self_attention
        
        return hidden_states[:, -1, :]  # [bz, out_dim]
    

class MemoryModel(nn.Module):
    def __init__(self, input_dim=20, trace_hiddens=[32, 64], trace_kernel_sizes=[3, 3], self_attn=True, chunk_length=30, trace_dropout=0.1, device='cpu', **kwargs):
        super(MemoryModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(input_dim, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, 
                           dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_length is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)
        

    def forward(self, x):  # [bz, T, input_dim]
        hidden_states = self.net(x)
        
        if self.self_attn: 
            hidden_states_after_self_attention = self.attn_layer(hidden_states)
           
            return hidden_states_after_self_attention
        
        return hidden_states[:, -1, :]  # [bz, out_dim]

class NetworkOutModel(nn.Module):
    def __init__(self, input_dim=20, trace_hiddens=[32, 64], trace_kernel_sizes=[3, 3], self_attn=True, chunk_length=30, trace_dropout=0.1, device='cpu', **kwargs):
        super(NetworkOutModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(input_dim, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, 
                           dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_length is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)
        

    def forward(self, x):  # [bz, T, input_dim]
        hidden_states = self.net(x)
        
        if self.self_attn: 
            hidden_states_after_self_attention = self.attn_layer(hidden_states)
            
            return hidden_states_after_self_attention
        
        return hidden_states[:, -1, :]  # [bz, out_dim]
    

class NetworkInModel(nn.Module):
    def __init__(self, input_dim=20, trace_hiddens=[32, 64], trace_kernel_sizes=[3, 3], self_attn=True, chunk_length=30, trace_dropout=0.1, device='cpu', **kwargs):
        super(NetworkInModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(input_dim, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, 
                           dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_length is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_length)
        

    def forward(self, x):  # [bz, T, input_dim]
        hidden_states = self.net(x)
        
        if self.self_attn: 
            hidden_states_after_self_attention = self.attn_layer(hidden_states)
            
            return hidden_states_after_self_attention
       
        return hidden_states[:, -1, :]  # [bz, out_dim]

class GraphModel(nn.Module):
    def __init__(self, in_dim, graph_hiddens=[64, 128], device='cpu', attn_head=4, activation=0.2, **kwargs):
        super(GraphModel, self).__init__()
        '''
        Params:
            in_dim: the feature dim of each node
        '''
        layers = []

        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i-1] if i > 0 else in_dim 
            dropout = kwargs["attn_drop"] if "attn_drop" in kwargs else 0
            layers.append(GATv2Conv(in_feats, out_feats=hidden, num_heads=attn_head, 
                                        attn_drop=dropout, negative_slope=activation, allow_zero_in_degree=True)) 
            self.maxpool = nn.MaxPool1d(attn_head)

        self.net = nn.Sequential(*layers).to(device)
        self.out_dim = graph_hiddens[-1]
        self.pooling = GlobalAttentionPooling(nn.Linear(self.out_dim, 1)) 

    
    def forward(self, graph, x):
        '''
        Input:
            x -- tensor float [batch_size*node_num, feature_in_dim] N = {s1, s2, s3, e1, e2, e3}
        '''
        out = None
        for layer in self.net:
            if out is None: out = x
            out = layer(graph, out)
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze()
        return self.pooling(graph, out) #[bz*node, out_dim] --> [bz, out_dim]


class MultiSourceEncoder(nn.Module):
    def __init__(self, node_num, device, log_dim=64, fuse_dim=64, alpha=0.1, **kwargs):
        super(MultiSourceEncoder, self).__init__()
        self.node_num = node_num
        self.alpha = alpha
        self.device = device
        self.low_level_dim = kwargs['input_dims']

        self.latency_model = LatencyModel(device=device, input_dim = self.low_level_dim, chunk_length=self.node_num, **kwargs)
        latency_dim = self.latency_model.out_dim

        self.cpu_model = CpuModel(device=device, input_dim = self.low_level_dim, chunk_length=self.node_num, **kwargs) 
        cpu_dim = self.cpu_model.out_dim

        self.memory_model = MemoryModel(device=device, input_dim = self.low_level_dim, chunk_length=self.node_num, **kwargs) 
        memory_dim = self.memory_model.out_dim

        self.networkout_model = NetworkOutModel(device=device, input_dim = self.low_level_dim, chunk_length=self.node_num, **kwargs) 
        networkout_dim = self.networkout_model.out_dim

        self.networkin_model = NetworkInModel(device=device, input_dim = self.low_level_dim, chunk_length=self.node_num, **kwargs) 
        networkin_dim = self.networkin_model.out_dim

        fuse_in = latency_dim + cpu_dim + memory_dim + networkout_dim + networkin_dim

        if not fuse_dim % 2 == 0: fuse_dim += 1
        self.fuse = nn.Linear(fuse_in, fuse_dim)

        self.activate = nn.GLU()
        self.feat_in_dim = int(fuse_dim // 2)

        self.status_model = GraphModel(in_dim=self.feat_in_dim, device=device, **kwargs)
        self.feat_out_dim = self.status_model.out_dim
    
    def forward(self, graph):
        latency_embedding = self.latency_model(graph.ndata['latency']) #[bz*node_num, T, trace_dim]
        latency_embedding = latency_embedding.reshape(-1, latency_embedding.size(2))

        cpu_embedding = self.cpu_model(graph.ndata["container_cpu_usage_seconds_total"]) #[bz*node_num, T, trace_dim]
        cpu_embedding = cpu_embedding.reshape(-1, cpu_embedding.size(2))

        memory_embedding = self.networkout_model(graph.ndata["container_memory_usage_bytes"]) #[bz*node_num, T, trace_dim]
        memory_embedding = memory_embedding.reshape(-1, memory_embedding.size(2))

        networkout_embedding = self.networkout_model(graph.ndata["container_network_transmit_bytes_total"]) #[bz*node_num, T, trace_dim]
        networkout_embedding = networkout_embedding.reshape(-1, networkout_embedding.size(2))

        networkin_embedding = self.networkin_model(graph.ndata["container_network_receive_bytes_total"]) #[bz*node_num, T, trace_dim]
        networkin_embedding = networkin_embedding.reshape(-1, networkin_embedding.size(2))


        feature = self.activate(self.fuse(torch.cat((latency_embedding, cpu_embedding, memory_embedding, 
                                                     networkout_embedding, networkin_embedding), dim=-1))) #[bz*node_num, node_dim]

        embeddings = self.status_model(graph, feature) #[bz, graph_dim]
        
        return embeddings


class MainModel(nn.Module):
    def __init__(self, node_num, device, alpha=0.1, **kwargs):
        super(MainModel, self).__init__()

        self.device = device
        self.node_num = node_num
        self.alpha = alpha

        self.encoder = MultiSourceEncoder(self.node_num, device, alpha=alpha, **kwargs)

        self.detector_criterion = nn.CrossEntropyLoss()
        self.localizer_criterion = nn.CrossEntropyLoss()

        self.detector = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)

        self.localizer1 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer2 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer3 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer4 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer5 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer6 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer7 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer8 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer9 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer10 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)


        self.localizer11 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer12 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer13 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer14 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer15 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer16 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer17 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer18 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer19 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer20 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)


        self.localizer21 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer22 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer23 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer24 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer25 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer26 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer27 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer28 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer29 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.localizer30 = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)


    def forward(self, graph, anomaly_gt, rootcause_gt):  
        batch_size = graph.batch_size
        embeddings = self.encoder(graph) #[bz, feat_out_dim]

        y_window_anomaly = torch.zeros(batch_size).long().to(self.device)

        y_local1_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local2_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local3_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local4_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local5_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local6_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local7_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local8_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local9_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local10_anomaly = torch.zeros(batch_size).long().to(self.device)

        y_local11_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local12_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local13_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local14_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local15_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local16_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local17_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local18_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local19_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local20_anomaly = torch.zeros(batch_size).long().to(self.device)

        y_local21_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local22_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local23_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local24_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local25_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local26_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local27_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local28_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local29_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_local30_anomaly = torch.zeros(batch_size).long().to(self.device)


        for i in range(batch_size):
            y_window_anomaly[i] = int(anomaly_gt[i])

            y_local1_anomaly[i] = int(rootcause_gt[i][0])
            y_local2_anomaly[i] = int(rootcause_gt[i][1])
            y_local3_anomaly[i] = int(rootcause_gt[i][2])
            y_local4_anomaly[i] = int(rootcause_gt[i][3])
            y_local5_anomaly[i] = int(rootcause_gt[i][4])
            y_local6_anomaly[i] = int(rootcause_gt[i][5])
            y_local7_anomaly[i] = int(rootcause_gt[i][6])
            y_local8_anomaly[i] = int(rootcause_gt[i][7])
            y_local9_anomaly[i] = int(rootcause_gt[i][8])
            y_local10_anomaly[i] = int(rootcause_gt[i][9])

            y_local11_anomaly[i] = int(rootcause_gt[i][10])
            y_local12_anomaly[i] = int(rootcause_gt[i][11])
            y_local13_anomaly[i] = int(rootcause_gt[i][12])
            y_local14_anomaly[i] = int(rootcause_gt[i][13])
            y_local15_anomaly[i] = int(rootcause_gt[i][14])
            y_local16_anomaly[i] = int(rootcause_gt[i][15])
            y_local17_anomaly[i] = int(rootcause_gt[i][16])
            y_local18_anomaly[i] = int(rootcause_gt[i][17])
            y_local19_anomaly[i] = int(rootcause_gt[i][18])
            y_local20_anomaly[i] = int(rootcause_gt[i][19])

            y_local21_anomaly[i] = int(rootcause_gt[i][20])
            y_local22_anomaly[i] = int(rootcause_gt[i][21])
            y_local23_anomaly[i] = int(rootcause_gt[i][22])
            y_local24_anomaly[i] = int(rootcause_gt[i][23])
            y_local25_anomaly[i] = int(rootcause_gt[i][24])
            y_local26_anomaly[i] = int(rootcause_gt[i][25])
            y_local27_anomaly[i] = int(rootcause_gt[i][26])
            y_local28_anomaly[i] = int(rootcause_gt[i][27])
            y_local29_anomaly[i] = int(rootcause_gt[i][28])
            y_local30_anomaly[i] = int(rootcause_gt[i][29])
            

        detect_logits = self.detector(embeddings)
        detect_loss = self.detector_criterion(detect_logits, y_window_anomaly) 

        locate1_logits = self.localizer1(embeddings)
        locate1_loss = self.localizer_criterion(locate1_logits, y_local1_anomaly)

        locate2_logits = self.localizer2(embeddings)
        locate2_loss = self.localizer_criterion(locate2_logits, y_local2_anomaly)

        locate3_logits = self.localizer3(embeddings)
        locate3_loss = self.localizer_criterion(locate3_logits, y_local3_anomaly)

        locate4_logits = self.localizer4(embeddings)
        locate4_loss = self.localizer_criterion(locate4_logits, y_local4_anomaly)

        locate5_logits = self.localizer5(embeddings)
        locate5_loss = self.localizer_criterion(locate5_logits, y_local5_anomaly)

        locate6_logits = self.localizer6(embeddings)
        locate6_loss = self.localizer_criterion(locate6_logits, y_local6_anomaly)

        locate7_logits = self.localizer7(embeddings)
        locate7_loss = self.localizer_criterion(locate7_logits, y_local7_anomaly)

        locate8_logits = self.localizer8(embeddings)
        locate8_loss = self.localizer_criterion(locate8_logits, y_local8_anomaly)

        locate9_logits = self.localizer9(embeddings)
        locate9_loss = self.localizer_criterion(locate9_logits, y_local9_anomaly)

        locate10_logits = self.localizer10(embeddings)
        locate10_loss = self.localizer_criterion(locate10_logits, y_local10_anomaly)


        locate11_logits = self.localizer11(embeddings)
        locate11_loss = self.localizer_criterion(locate11_logits, y_local11_anomaly)

        locate12_logits = self.localizer12(embeddings)
        locate12_loss = self.localizer_criterion(locate12_logits, y_local12_anomaly)

        locate13_logits = self.localizer13(embeddings)
        locate13_loss = self.localizer_criterion(locate13_logits, y_local13_anomaly)

        locate14_logits = self.localizer14(embeddings)
        locate14_loss = self.localizer_criterion(locate14_logits, y_local14_anomaly)

        locate15_logits = self.localizer15(embeddings)
        locate15_loss = self.localizer_criterion(locate15_logits, y_local15_anomaly)

        locate16_logits = self.localizer16(embeddings)
        locate16_loss = self.localizer_criterion(locate16_logits, y_local16_anomaly)

        locate17_logits = self.localizer17(embeddings)
        locate17_loss = self.localizer_criterion(locate17_logits, y_local17_anomaly)

        locate18_logits = self.localizer18(embeddings)
        locate18_loss = self.localizer_criterion(locate18_logits, y_local18_anomaly)

        locate19_logits = self.localizer19(embeddings)
        locate19_loss = self.localizer_criterion(locate19_logits, y_local19_anomaly)

        locate20_logits = self.localizer20(embeddings)
        locate20_loss = self.localizer_criterion(locate20_logits, y_local20_anomaly)


        locate21_logits = self.localizer21(embeddings)
        locate21_loss = self.localizer_criterion(locate21_logits, y_local21_anomaly)

        locate22_logits = self.localizer22(embeddings)
        locate22_loss = self.localizer_criterion(locate22_logits, y_local22_anomaly)

        locate23_logits = self.localizer23(embeddings)
        locate23_loss = self.localizer_criterion(locate23_logits, y_local23_anomaly)

        locate24_logits = self.localizer24(embeddings)
        locate24_loss = self.localizer_criterion(locate24_logits, y_local24_anomaly)

        locate25_logits = self.localizer25(embeddings)
        locate25_loss = self.localizer_criterion(locate25_logits, y_local25_anomaly)

        locate26_logits = self.localizer26(embeddings)
        locate26_loss = self.localizer_criterion(locate26_logits, y_local26_anomaly)

        locate27_logits = self.localizer27(embeddings)
        locate27_loss = self.localizer_criterion(locate27_logits, y_local27_anomaly)

        locate28_logits = self.localizer28(embeddings)
        locate28_loss = self.localizer_criterion(locate28_logits, y_local28_anomaly)

        locate29_logits = self.localizer29(embeddings)
        locate29_loss = self.localizer_criterion(locate29_logits, y_local29_anomaly)

        locate30_logits = self.localizer30(embeddings)
        locate30_loss = self.localizer_criterion(locate30_logits, y_local30_anomaly)


        loss = self.alpha * detect_loss + (1-self.alpha)/30 * locate1_loss + (1-self.alpha)/30 * locate2_loss + (1-self.alpha)/30 * locate3_loss + (1-self.alpha)/30 * locate4_loss + (1-self.alpha)/30 * locate5_loss + (1-self.alpha)/30 * locate6_loss + (1-self.alpha)/30 * locate7_loss + (1-self.alpha)/30 * locate8_loss + (1-self.alpha)/30 * locate9_loss + (1-self.alpha)/30 * locate10_loss \
        + (1-self.alpha)/30 * locate11_loss + (1-self.alpha)/30 * locate12_loss + (1-self.alpha)/30 * locate13_loss + (1-self.alpha)/30 * locate14_loss + (1-self.alpha)/30 * locate15_loss + (1-self.alpha)/30 * locate16_loss + (1-self.alpha)/30 * locate17_loss + (1-self.alpha)/30 * locate18_loss + (1-self.alpha)/30 * locate19_loss + (1-self.alpha)/30 * locate20_loss \
        + (1-self.alpha)/30 * locate21_loss + (1-self.alpha)/30 * locate22_loss + (1-self.alpha)/30 * locate23_loss + (1-self.alpha)/30 * locate24_loss + (1-self.alpha)/30 * locate25_loss + (1-self.alpha)/30 * locate26_loss + (1-self.alpha)/30 * locate27_loss + (1-self.alpha)/30 * locate28_loss + (1-self.alpha)/30 * locate29_loss + (1-self.alpha)/30 * locate30_loss       


        graph_logits, \
            l1_logits, l2_logits, l3_logits, l4_logits, l5_logits, l6_logits, l7_logits, l8_logits, l9_logits, l10_logits, \
            l11_logits, l12_logits, l13_logits, l14_logits, l15_logits, l16_logits, l17_logits, l18_logits, l19_logits, l20_logits, \
            l21_logits, l22_logits, l23_logits, l24_logits, l25_logits, l26_logits, l27_logits, l28_logits, l29_logits, l30_logits = self.inference(batch_size, 
                                                                            detect_logits, 
                                                                            locate1_logits, 
                                                                            locate2_logits,
                                                                            locate3_logits,
                                                                            locate4_logits,
                                                                            locate5_logits,
                                                                            locate6_logits, 
                                                                            locate7_logits,
                                                                            locate8_logits,
                                                                            locate9_logits,
                                                                            locate10_logits,
                                                                            locate11_logits, 
                                                                            locate12_logits,
                                                                            locate13_logits,
                                                                            locate14_logits,
                                                                            locate15_logits,
                                                                            locate16_logits, 
                                                                            locate17_logits,
                                                                            locate18_logits,
                                                                            locate19_logits,
                                                                            locate20_logits,
                                                                            locate21_logits, 
                                                                            locate22_logits,
                                                                            locate23_logits,
                                                                            locate24_logits,
                                                                            locate25_logits,
                                                                            locate26_logits, 
                                                                            locate27_logits,
                                                                            locate28_logits,
                                                                            locate29_logits,
                                                                            locate30_logits)

        return {'loss': loss, 
                'graph_logits': graph_logits, 
                'l1_logits': l1_logits, 
                'l2_logits': l2_logits,
                'l3_logits': l3_logits,
                'l4_logits': l4_logits,
                'l5_logits': l5_logits,
                'l6_logits': l6_logits,
                'l7_logits': l7_logits,
                'l8_logits': l8_logits,
                'l9_logits': l9_logits,
                'l10_logits': l10_logits,
                'l11_logits': l11_logits, 
                'l12_logits': l12_logits,
                'l13_logits': l13_logits,
                'l14_logits': l14_logits,
                'l15_logits': l15_logits,
                'l16_logits': l16_logits,
                'l17_logits': l17_logits,
                'l18_logits': l18_logits,
                'l19_logits': l19_logits,
                'l20_logits': l20_logits,
                'l21_logits': l21_logits, 
                'l22_logits': l22_logits,
                'l23_logits': l23_logits,
                'l24_logits': l24_logits,
                'l25_logits': l25_logits,
                'l26_logits': l26_logits,
                'l27_logits': l27_logits,
                'l28_logits': l28_logits,
                'l29_logits': l29_logits,
                'l30_logits': l30_logits}

    def inference(self, batch_size, detect_logits=None, locate1_logits=None, locate2_logits=None, locate3_logits=None, locate4_logits=None, locate5_logits=None, locate6_logits=None, locate7_logits=None, locate8_logits=None, locate9_logits=None, locate10_logits=None, , locate11_logits=None, , locate12_logits=None, locate13_logits=None, locate14_logits=None, locate3_logits=None):
        
        # for i in range(batch_size):
        detect_pred = detect_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        predictions = torch.tensor(detect_pred)


        locate1_logits = locate1_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        l1_predictions = torch.tensor(locate1_logits)


        locate2_logits = locate2_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        l2_predictions = torch.tensor(locate2_logits)

        locate3_logits = locate3_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        l3_predictions = torch.tensor(locate3_logits)

        locate4_logits = locate4_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        l4_predictions = torch.tensor(locate4_logits)

        locate5_logits = locate5_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
        l5_predictions = torch.tensor(locate5_logits)

        return predictions, l1_predictions, l2_predictions, l3_predictions, l4_predictions, l5_predictions