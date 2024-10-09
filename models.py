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
import random

import pdb 

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

import math
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
        # x: [batch_size, window_size, input_size]
        input_tensor = x.transpose(1, 0)  # w x b x h
        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
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
        self.chunk_length = kwargs['chunk_length']

        self.latency_model = LatencyModel(device=device, input_dim = self.low_level_dim, **kwargs)
        latency_dim = self.latency_model.out_dim

        self.cpu_model = CpuModel(device=device, input_dim = self.low_level_dim, **kwargs) 
        cpu_dim = self.cpu_model.out_dim

        self.memory_model = MemoryModel(device=device, input_dim = self.low_level_dim, **kwargs) 
        memory_dim = self.memory_model.out_dim

        self.networkout_model = NetworkOutModel(device=device, input_dim = self.low_level_dim, **kwargs) 
        networkout_dim = self.networkout_model.out_dim

        self.networkin_model = NetworkInModel(device=device, input_dim = self.low_level_dim, **kwargs) 
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
        #latency_embedding = latency_embedding.reshape(-1, latency_embedding.size(2))

        cpu_embedding = self.cpu_model(graph.ndata["container_cpu_usage_seconds_total"]) #[bz*node_num, T, trace_dim]
        #cpu_embedding = cpu_embedding.reshape(-1, cpu_embedding.size(2))

        memory_embedding = self.networkout_model(graph.ndata["container_memory_usage_bytes"]) #[bz*node_num, T, trace_dim]
        #memory_embedding = memory_embedding.reshape(-1, memory_embedding.size(2))

        networkout_embedding = self.networkout_model(graph.ndata["container_network_transmit_bytes_total"]) #[bz*node_num, T, trace_dim]
        #networkout_embedding = networkout_embedding.reshape(-1, networkout_embedding.size(2))

        networkin_embedding = self.networkin_model(graph.ndata["container_network_receive_bytes_total"]) #[bz*node_num, T, trace_dim]
        #networkin_embedding = networkin_embedding.reshape(-1, networkin_embedding.size(2))


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
        self.weight_loss = kwargs['weight_loss']
        self.encoder = MultiSourceEncoder(self.node_num, device, alpha=alpha, **kwargs)
        
        self.detector_criterion = nn.CrossEntropyLoss(torch.FloatTensor(self.weight_loss).to(self.device))
        self.detector = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.get_prob = nn.Softmax(dim=-1)


    def forward(self, graph, anomaly_gt, rootcause_gt, only_train=False):  
        batch_size = graph.batch_size
        embeddings = self.encoder(graph) #[bz, feat_out_dim]

        y_window_anomaly = torch.zeros(batch_size).long().to(self.device)
        y_node_anomaly = torch.zeros(batch_size, self.node_num).long().to(self.device)

        for i in range(batch_size):
            y_window_anomaly[i] = int(anomaly_gt[i])
            y_node_anomaly[i, :] = rootcause_gt[i]

        detect_logits = self.detector(embeddings)
        loss = self.detector_criterion(detect_logits, y_window_anomaly)
        prob_logits = self.get_prob(detect_logits) 

        if only_train == False:
            y_pred, max_score_diff, gt_score_diff = self.inference(batch_size, graph, prob_logits, rootcause_gt)
            return {'loss': loss,'y_pred': y_pred, 'y_prob': y_node_anomaly.detach().cpu().numpy(), 'max_score_diff': max_score_diff, 'gt_score_diff': gt_score_diff }

        else:
            return {'loss': loss}


    def inference(self, batch_size, graph, prob_logits, rootcause_gt):

        y_pred = []
        max_score_diff = []
        gt_score_diff = []

        detect_pred = prob_logits.argmax(axis=1).squeeze() 
        graph_list = dgl.unbatch(graph)
        
        for i in range(batch_size):
            if detect_pred[i] < 1: 
                y_pred.append([-1]) #anomaly 가 없으면 -1 값 
                max_score_diff.append(-1)
                gt_score_diff.append(-1)

            else: #anomaly 가 있다면?
                # anomaly_indexs.append(i)
                max_diff = -1  
                best_mask = None 
                gt_mask = [1 - x for x in rootcause_gt[i]]
                current_graph = graph_list[i]

                rootcause_number = (rootcause_gt[i]==1).sum().item()
                
                if rootcause_number == 0:
                    gt_score_diff.append(-1)
                    
                else:
                    gt_masked_graph = self.graph_masking(current_graph, gt_mask) 
                    gt_masked_embeddings = self.encoder(gt_masked_graph)
                    gt_masked_detect_logit = self.detector(gt_masked_embeddings)
                    gt_masked_prob_logit = self.get_prob(gt_masked_detect_logit)
                    gt_diff = prob_logits[i][1] - gt_masked_prob_logit[0][1]
                    gt_score_diff.append(gt_diff)

                for _ in range(10): #랜덤 마스크 10번 적용 
                    if rootcause_number == 0:
                        virtual_rootcause_number = 1
                    else:
                        virtual_rootcause_number = rootcause_number

                    random_mask = [0] * virtual_rootcause_number + [1] * (self.node_num - virtual_rootcause_number)   #random_mask = [random.choice([0, 1]) for _ in range(k)] #0과 1로만 이루어짐 
                    random.shuffle(random_mask) 

                    masked_graph = self.graph_masking(current_graph, random_mask) 
                    masked_embeddings = self.encoder(masked_graph)
                    masked_detect_logit = self.detector(masked_embeddings)
                    masked_prob_logit = self.get_prob(masked_detect_logit)

                    diff = prob_logits[i][1] - masked_prob_logit[0][1] #original anomaly 확률 값 - masked anomaly 확률값의 차이, 이것이 클수록 mask된 노드가 RC일 확률 높음
                    if diff > max_diff:
                        max_diff = diff
                        best_mask = random_mask 

                #best_mask에서 값이 0 인 노드들이 rootcause일 것이다. 
                rootcause_indexs = [index for index, value in enumerate(best_mask) if value == 0]
                y_pred.append(rootcause_indexs)
                max_score_diff.append(max_diff)

        return y_pred, max_score_diff, gt_score_diff

    def graph_masking(self, graph, node_mask):

        unmasked_index = [] 
        for i in range(self.node_num):
            if node_mask[i] == 1:
                unmasked_index.append(i)

        subgraph = dgl.node_subgraph(graph, unmasked_index)

        return subgraph 

#     def inference(self, batch_size, graph, prob_logits):

#         y_pred = []
#         detect_pred = prob_logits.argmax(axis=1).squeeze() 
#         diff_lists = [] 
#         for i in range(batch_size):
#             graph_list = dgl.unbatch(graph)

#             if detect_pred[i] < 1: 
#                 y_pred.append([-1]) #anomaly 가 없으면 -1 값 
#                 diff_lists.append([-1])

#             else: #anomaly 가 있다면?
#                 # anomaly_indexs.append(i)
#                 diff_list = []
#                 current_graph = graph_list[i]
#                 for j in range(self.node_num): # node 하나씩 제거해보기 
#                     masked_graph = self.masking(current_graph, j) #노드 하나씩 제거하는 코드
#                     masked_embeddings = self.encoder(masked_graph)
#                     masked_detect_logit = self.detector(masked_embeddings)
#                     masked_prob_logit = self.get_prob(masked_detect_logit)

#                     diff = prob_logits[i][1] - masked_prob_logit[0][1] #original anomaly 확률 값 - masked anomaly 확률값의 차이, 이것이 클수록 mask된 노드가 RC일 확률 높음
#                     diff_list.append(diff.item())

#                 diff_lists.append(diff_list)

            

#                 temp = np.argsort(diff_list)[::-1]
#                 # diff_lists.append(diff_list)
#                 y_pred.append(temp)


#         #return y_pred, diff_lists, anomaly_indexs
#         return y_pred, diff_lists 

# # j 번째 node 삭제와, 해당 node와 관련된 edge들까지 삭제해야 함 
#     def masking(self, graph, node_index):
#         #j 번째 node 삭제 
#         unmaksed_index = list(range(self.node_num))
#         unmaksed_index.remove(node_index) 

#         # 남은 노드들로부터 새로운 서브그래프를 생성합니다.
#         subgraph = dgl.node_subgraph(graph, unmaksed_index)

#         return subgraph
