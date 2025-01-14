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
    def __init__(self, in_dim, graph_hiddens=[64, 128], device='cpu', attn_head=4, activation=0.2, **kwargs): #hidden layer 를
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

        self.activate = nn.GLU() # GLU는 출력차원이 절반으로 줄어듦 
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
    def __init__(self, node_num, device, normal_avg, alpha=0.1, **kwargs):
        super(MainModel, self).__init__()

        self.device = device
        self.node_num = node_num
        self.alpha = alpha
        self.weight_loss = kwargs['weight_loss'] # loss 함수에 weight를 줘서 데이터 불균형을 맞추기 위함
        self.encoder = MultiSourceEncoder(self.node_num, device, alpha=alpha, **kwargs)
        self.normal_avg = normal_avg

        self.detector_criterion = nn.CrossEntropyLoss()
        #self.detector_criterion = nn.CrossEntropyLoss(torch.FloatTensor(self.weight_loss).to(device))
        self.detector = FullyConnected(self.encoder.feat_out_dim, 2, [64, 64]).to(device)
        self.get_prob = nn.Softmax(dim=-1)

        self.call_paths = kwargs['call_paths']
        self.placements = kwargs['placements']


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


        #normal inference(node masking)
        if only_train == False: #test 시에는 Rootcause localization task에 대한 성능 평가를 위해 RC inference를 해야 함 
            y_pred, rootcause_avg_diff, normal_avg_diff  = self.inference(graph, prob_logits, rootcause_gt)
            random_y_pred =  self.random_inference(graph, prob_logits, rootcause_gt)
            return {'loss': loss,'y_pred': y_pred, 'random_y_pred': random_y_pred, 'rootcause_avg_diff': rootcause_avg_diff,'normal_avg_diff': normal_avg_diff}          

        # advanced inference(multi level masking)
        # if only_train == False:
        #     y_pred  = self.advanced_inference2(graph, prob_logits, rootcause_gt)
        #     random_y_pred =  self.random_inference(graph, prob_logits, rootcause_gt)
        #     return {'loss': loss,'y_pred': y_pred, 'random_y_pred': random_y_pred}   

        #train 시에는 anomaly detection task에 대해서만 학습하면 됨
        else:
            return {'loss': loss}


    def inference(self,graph, prob_logits, rootcause_gt):
        batch_size = graph.batch_size

        y_pred = []

        ##### for prelimanary experiment ###################
        rootcause_avg_diff = 0 
        normal_avg_diff = 0

        #####################################################


        detect_pred = prob_logits.argmax(axis=1).squeeze() 
        graph_list = dgl.unbatch(graph)
        
        for i in range(batch_size):
            if detect_pred[i] < 1: 
                y_pred.append([-1]) #anomaly 가 없으면 -1 값 

            else: #anomaly 가 있다면?
                current_graph = graph_list[i]

                masked_graph_list, _ = self.generate_masked_graphs(current_graph, "node")
                masked_graph_batch = dgl.batch(masked_graph_list)

                masked_embeddings = self.encoder(masked_graph_batch)
                masked_detect_logit = self.detector(masked_embeddings)
                masked_prob_logit = self.get_prob(masked_detect_logit)
                
                original_prob_logits = prob_logits[i].repeat(self.node_num,1)

                diff_list = original_prob_logits[:, 1] - masked_prob_logit[ : ,1] #original anomaly 확률 값 - masked anomaly 확률값의 차이, 이것이 클수록 mask된 노드가 RC일 확률 높음

                ##### for prelimanary experiment #######################################################
                if sum(rootcause_gt[i]) != 0: # TP일때 
                    rootcause_indices = [i for i, value in enumerate(rootcause_gt[i]) if value == 1]
                    normal_indices = [i for i, value in enumerate(rootcause_gt[i]) if value == 0]
                    rootcause_diff = sum([diff_list[i] for i in rootcause_indices]) / len(rootcause_indices)
                    normal_diff = sum([diff_list[i] for i in normal_indices]) / len(normal_indices)
                    

                    rootcause_avg_diff += rootcause_diff
                    normal_avg_diff += normal_diff
                #########################################################################################


                ranked_list = sorted(range(len(diff_list)), key=lambda i: diff_list[i], reverse=True)

                y_pred.append(ranked_list)

        ##### for prelimanary experiment ###################
        rootcause_avg_diff = rootcause_avg_diff / batch_size
        normal_avg_diff = normal_avg_diff / batch_size 

        #####################################################
    
        return y_pred, rootcause_avg_diff, normal_avg_diff 


    def advanced_inference(self,graph, prob_logits, rootcause_gt):
        batch_size = graph.batch_size

        y_pred = []

        detect_pred = prob_logits.argmax(axis=1).squeeze() 
        graph_list = dgl.unbatch(graph)
        
        for i in range(batch_size):
            if detect_pred[i] < 1: 
                y_pred.append([-1]) #anomaly 가 없으면 -1 값 

            else: #anomaly 가 있다면?
                rootcause_scores = [0 for _ in range(self.node_num)] #여기에 3가지 masking 기법을 통해 얻어진 anomaly score diff 가 integrated 됨 
                current_graph = graph_list[i]

                # placement masking 
                masked_graph_list, masked_node_lists = self.generate_masked_graphs(current_graph, "place") 
                masked_graph_batch = dgl.batch(masked_graph_list)

                masked_embeddings = self.encoder(masked_graph_batch)
                masked_detect_logit = self.detector(masked_embeddings)
                masked_prob_logit = self.get_prob(masked_detect_logit)
                
                original_prob_logits = prob_logits[i].repeat(len(self.placements),1)

                diff_list = original_prob_logits[:, 1] - masked_prob_logit[ : ,1] #original anomaly 확률 값 - masked anomaly 확률값의 차이, 이것이 클수록 mask된 노드가 RC일 확률 높음

                for i in range(len(self.placements)):
                    node_indexs = masked_node_lists[i]
                    for node_index in node_indexs:
                        rootcause_scores[node_index] += diff_list[i]

                # path masking, 중복되는 노드들이 있으므로 avg 로 해야함 
                # masked_graph_list, masked_node_lists = self.generate_masked_graphs(current_graph, "path") 
                # masked_graph_batch = dgl.batch(masked_graph_list)

                # masked_embeddings = self.encoder(masked_graph_batch)
                # masked_detect_logit = self.detector(masked_embeddings)
                # masked_prob_logit = self.get_prob(masked_detect_logit)
                
                # original_prob_logits = prob_logits[i].repeat(len(self.call_paths),1)

                # diff_list = original_prob_logits[:, 1] - masked_prob_logit[ : ,1] #original anomaly 확률 값 - masked anomaly 확률값의 차이, 이것이 클수록 mask된 노드가 RC일 확률 높음

                # temp = [[] for _ in range(self.node_num)]

                # for i in range(len(self.call_paths)):
                #     node_indexs = masked_node_lists[i]
                #     for node_index in node_indexs: # path에 포함되는 노드들 순회
                #         temp[node_index].append(diff_list[i].item())

                # for node_index in range(self.node_num):
                #     values = temp[node_index]
                #     rootcause_scores[node_index] += np.mean(values)

                # node masking
                # masked_graph_list, masked_node_lists = self.generate_masked_graphs(current_graph, "node") 
                # masked_graph_batch = dgl.batch(masked_graph_list)

                # masked_embeddings = self.encoder(masked_graph_batch)
                # masked_detect_logit = self.detector(masked_embeddings)
                # masked_prob_logit = self.get_prob(masked_detect_logit)
                
                # original_prob_logits = prob_logits[i].repeat(self.node_num,1)

                # diff_list = original_prob_logits[:, 1] - masked_prob_logit[ : ,1] #original anomaly 확률 값 - masked anomaly 확률값의 차이, 이것이 클수록 mask된 노드가 RC일 확률 높음

                # for i in range(self.node_num):
                #     node_indexs = masked_node_lists[i]
                #     for node_index in node_indexs:
                #         rootcause_scores[node_index] += diff_list[i]


                #get ranked list from rootcause score
                temp = sorted(list(enumerate(rootcause_scores)), key=lambda x: x[1], reverse=True)
                ranked_list = [index for index, _ in temp]

                y_pred.append(ranked_list)

        return y_pred

    def advanced_inference2(self,graph, prob_logits, rootcause_gt):
        batch_size = graph.batch_size

        y_pred = []

        detect_pred = prob_logits.argmax(axis=1).squeeze() 
        graph_list = dgl.unbatch(graph)
        
        for i in range(batch_size):
            if detect_pred[i] < 1: 
                y_pred.append([-1]) #anomaly 가 없으면 -1 값 

            else: #anomaly 가 있다면?
                rootcause_scores = [0 for _ in range(self.node_num)] 
                current_graph = graph_list[i]


                # node masking
                masked_graph_list, masked_node_lists = self.generate_masked_graphs(current_graph, "node") 
                masked_graph_batch = dgl.batch(masked_graph_list)

                masked_embeddings = self.encoder(masked_graph_batch)
                masked_detect_logit = self.detector(masked_embeddings)
                masked_prob_logit = self.get_prob(masked_detect_logit)
                
                original_prob_logits = prob_logits[i].repeat(self.node_num,1)

                diff_list = original_prob_logits[:, 1] - masked_prob_logit[ : ,1] #original anomaly 확률 값 - masked anomaly 확률값의 차이, 이것이 클수록 mask된 노드가 RC일 확률 높음

                for i in range(len(self.placements)):
                    node_indexs = masked_node_lists[i]
                    max_value = -1 
                    for node_index in node_indexs:
                        if diff_list[node_index] > max_value:
                            max_value = diff_list[node_index]
                    for node_index in node_indexs:
                        rootcause_scores[node_index] = max_value

                #get ranked list from rootcause score
                temp = sorted(list(enumerate(rootcause_scores)), key=lambda x: x[1], reverse=True)
                ranked_list = [index for index, _ in temp]

                y_pred.append(ranked_list)

        return y_pred


    def random_inference(self,graph, prob_logits, rootcause_gt):
        batch_size = graph.batch_size

        y_pred = []

        detect_pred = prob_logits.argmax(axis=1).squeeze() 
        
        for i in range(batch_size):
            if detect_pred[i] < 1: 
                y_pred.append([-1]) #anomaly 가 없으면 -1 값 

            else: #anomaly 가 있다면?
                random_ranked_list = list(range(self.node_num))
                random.shuffle(random_ranked_list)
                y_pred.append(random_ranked_list)

        return y_pred


#병렬 처리를 위함. 
    def generate_masked_graphs(self, graph, mask_type="node"):
        masked_graphs = [] # 마스킹 된 그래프 리스트 
        masked_node_lists = [] 

        if mask_type == "node":
            for i in range(self.node_num):
                masked_graph = self.node_masking(graph, i)
                masked_graphs.append(masked_graph)
                temp = []
                temp.append(i)
                masked_node_lists.append(temp)

        elif mask_type == "path":
            for path_index in self.call_paths.keys():
                node_indexs = self.call_paths[path_index]
                masked_graph = self.path_masking(graph,node_indexs)
                masked_graphs.append(masked_graph)
                masked_node_lists.append(node_indexs)

        else: #mask_type = place 
            for vm_index in self.placements.keys():
                node_indexs = self.placements[vm_index]
                masked_graph = self.place_masking(graph, node_indexs)
                masked_graphs.append(masked_graph)
                masked_node_lists.append(node_indexs)
            
        return masked_graphs, masked_node_lists


    def node_masking(self, graph, node_index):

        unmaksed_index = list(range(self.node_num))
        unmaksed_index.remove(node_index) 

        # 남은 노드들로부터 새로운 서브그래프를 생성합니다.
        subgraph = dgl.node_subgraph(graph, unmaksed_index)

        return subgraph

#해당 call path 에 있는 마이크로서비스들 subgraph 단위로 masking 
    def path_masking(self, graph, node_indexs):

        unmaksed_index = [i for i in range(self.node_num) if i not in node_indexs]

        # 남은 노드들로부터 새로운 서브그래프를 생성합니다.
        subgraph = dgl.node_subgraph(graph, unmaksed_index)

        return subgraph

# vm 에 deploy된 마이크로서비스들 함께 masking 
    def place_masking(self, graph, node_indexs):

        unmaksed_index = [i for i in range(self.node_num) if i not in node_indexs]

        # 남은 노드들로부터 새로운 서브그래프를 생성합니다.
        subgraph = dgl.node_subgraph(graph, unmaksed_index)

        return subgraph


    def zero_masking(self, graph, node_index):
        subgraph = graph.clone()

        subgraph.ndata['latency'] = copy.deepcopy(graph.ndata['latency'])
        subgraph.ndata['container_cpu_usage_seconds_total'] = copy.deepcopy(graph.ndata['container_cpu_usage_seconds_total'])
        subgraph.ndata['container_network_transmit_bytes_total'] = copy.deepcopy(graph.ndata['container_network_transmit_bytes_total'])
        subgraph.ndata['container_network_receive_bytes_total'] = copy.deepcopy(graph.ndata['container_network_receive_bytes_total'])
        subgraph.ndata['container_memory_usage_bytes'] = copy.deepcopy(graph.ndata['container_memory_usage_bytes'])

        # 변경하고자 하는 노드의 데이터를 업데이트
        subgraph.ndata['latency'][node_index] = torch.zeros(subgraph.ndata['latency'][node_index].shape)
        subgraph.ndata['container_cpu_usage_seconds_total'][node_index] = torch.zeros(subgraph.ndata['container_cpu_usage_seconds_total'][node_index].shape)
        subgraph.ndata['container_network_transmit_bytes_total'][node_index] = torch.zeros(subgraph.ndata['container_network_transmit_bytes_total'][node_index].shape)
        subgraph.ndata['container_network_receive_bytes_total'][node_index] = torch.zeros(subgraph.ndata['container_network_receive_bytes_total'][node_index].shape)
        subgraph.ndata['container_memory_usage_bytes'][node_index] = torch.zeros(subgraph.ndata['container_memory_usage_bytes'][node_index].shape)

        return subgraph

    def normal_masking(self, graph, node_index):
        subgraph = graph.clone()

        subgraph.ndata['latency'] = copy.deepcopy(graph.ndata['latency'])
        subgraph.ndata['container_cpu_usage_seconds_total'] = copy.deepcopy(graph.ndata['container_cpu_usage_seconds_total'])
        subgraph.ndata['container_network_transmit_bytes_total'] = copy.deepcopy(graph.ndata['container_network_transmit_bytes_total'])
        subgraph.ndata['container_network_receive_bytes_total'] = copy.deepcopy(graph.ndata['container_network_receive_bytes_total'])
        subgraph.ndata['container_memory_usage_bytes'] = copy.deepcopy(graph.ndata['container_memory_usage_bytes'])

        # 변경하고자 하는 노드의 데이터를 업데이트
        subgraph.ndata['latency'][node_index] = self.normal_avg['avg_latency'][node_index].clone() 
        subgraph.ndata['container_cpu_usage_seconds_total'][node_index] = self.normal_avg['avg_cpu'][node_index].clone() 
        subgraph.ndata['container_network_transmit_bytes_total'][node_index] = self.normal_avg['avg_network_out'][node_index].clone() 
        subgraph.ndata['container_network_receive_bytes_total'][node_index] = self.normal_avg['avg_network_in'][node_index].clone() 
        subgraph.ndata['container_memory_usage_bytes'][node_index] = self.normal_avg['avg_memory'][node_index].clone() 

        return subgraph

    def neighbor_masking(self, graph, node_index):
        subgraph = graph.clone()

        subgraph.ndata['latency'] = copy.deepcopy(graph.ndata['latency'])
        subgraph.ndata['container_cpu_usage_seconds_total'] = copy.deepcopy(graph.ndata['container_cpu_usage_seconds_total'])
        subgraph.ndata['container_network_transmit_bytes_total'] = copy.deepcopy(graph.ndata['container_network_transmit_bytes_total'])
        subgraph.ndata['container_network_receive_bytes_total'] = copy.deepcopy(graph.ndata['container_network_receive_bytes_total'])
        subgraph.ndata['container_memory_usage_bytes'] = copy.deepcopy(graph.ndata['container_memory_usage_bytes'])

        # 변경하고자 하는 노드의 데이터를 업데이트
        subgraph.ndata['latency'][node_index] = self.normal_avg['avg_latency'][node_index].clone() 
        subgraph.ndata['container_cpu_usage_seconds_total'][node_index] = self.normal_avg['avg_cpu'][node_index].clone() 
        subgraph.ndata['container_network_transmit_bytes_total'][node_index] = self.normal_avg['avg_network_out'][node_index].clone() 
        subgraph.ndata['container_network_receive_bytes_total'][node_index] = self.normal_avg['avg_network_in'][node_index].clone() 
        subgraph.ndata['container_memory_usage_bytes'][node_index] = self.normal_avg['avg_memory'][node_index].clone() 

        return subgraph