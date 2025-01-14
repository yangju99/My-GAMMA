from torch.utils.data import Dataset, DataLoader
import torch
import dgl
from utils import * #collate, get_device 등의 함수 import 
import pickle
import sys
import logging
from base import BaseModel
import time
from utils import *
import pandas as pd 
import pdb 

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class chunkDataset(Dataset): #[node_num, T]
    def __init__(self, chunks, edges, num_node):
        self.data = []
        self.idx2id = {}
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk["window_id"]
            self.idx2id[idx] = chunk_id
            graph = dgl.graph(edges, num_nodes = num_node) #changed!
            graph = dgl.add_self_loop(graph) #zero in-degree nodes will lead to invalid output value

            
            # (node_num, window_size)
            graph.ndata["latency"] = torch.FloatTensor(chunk["latency"])
            graph.ndata["container_cpu_usage_seconds_total"] = torch.FloatTensor(chunk["container_cpu_usage_seconds_total"])
            graph.ndata["container_network_transmit_bytes_total"] = torch.FloatTensor(chunk["container_network_transmit_bytes_total"])
            graph.ndata["container_memory_usage_bytes"] = torch.FloatTensor(chunk["container_memory_usage_bytes"])
            graph.ndata["container_network_receive_bytes_total"] = torch.FloatTensor(chunk["container_network_receive_bytes_total"])

            anomaly_gt = chunk['label_window']
            rootcause_gt = np.zeros(num_node)
            for i in range(num_node):
                rootcause_gt[i] = chunk["label_" + str(i)]  

            self.data.append((graph, anomaly_gt, rootcause_gt))
                
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __get_chunk_id__(self, idx):
        return self.idx2id[idx]


def parse_call_paths(file_path):
    call_paths = {}
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            path = list(map(int, line.strip().split('->')))
            call_paths[index] = path

    return call_paths

def parse_placements(placement_file):
    place_df = pd.read_csv(placement_file, header=None)
    place_df.columns = ['microservice_index', 'service_name', 'vm_index', 'node_type']

    vm_mapping = place_df.groupby('vm_index')['microservice_index'].apply(list).to_dict()
    
    return vm_mapping

def run(params):

    call_path_file = params['call_path_file']
    call_paths = parse_call_paths(call_path_file)
    params['call_paths'] = call_paths 
    
    placement_file = params['placement_file']
    placements = parse_placements(placement_file)
    params['placements'] = placements

    train_file = params["train_file"]
    with open(train_file, 'rb') as tr:
        train_chunks = pickle.load(tr)
    tr.close()

    test_file = params["test_file"]
    with open(test_file, 'rb') as te:
        test_chunks = pickle.load(te)
    te.close()

    hash_id = dump_params(params)

    params["hash_id"] = hash_id

    # loss 함수에 weight를 반영하기 위함, anomlay 와 normal 데이터 셋 비율을 맞추는 대신 loss 함수 가중치를 통해 하려고 했음 
    num_traindata = len(train_chunks)
    anomaly_cnt_traindata = 0 
    for chunk in train_chunks:
        if chunk['label_window'] == 1:
            anomaly_cnt_traindata += 1
    normal_cnt_traindata = num_traindata - anomaly_cnt_traindata
    normed_weights = [1-(normal_cnt_traindata/num_traindata), 1-(anomaly_cnt_traindata/num_traindata)]

    params['weight_loss'] = normed_weights


    edges_set = set()
    # edge 정보 load하기 
    # 1. placement 에서 physical relationship(edge)정보 추가 (양방향) 
    # for key, nodes in placements.items():
    #     for i in range(len(nodes)):
    #         for j in range(i+1, len(nodes)):
    #             edges_set.add((nodes[i], nodes[j]))
    #             edges_set.add((nodes[j], nodes[i]))
        
    # # 2. call path 에서 logical relationship(edge)정보 추가 (양방향)
    for path in call_paths.values():
        for i in range(len(path)-1):
            edges_set.add((path[i], path[i + 1]))
            edges_set.add((path[i + 1], path[i])) 

    

    source, target = zip(*edges_set)
    edges = (source, target)

    train_data = chunkDataset(train_chunks, edges, params['nodes'])
    test_data = chunkDataset(test_chunks, edges, params['nodes'])


    normal_avg = extract_normal_status(train_data)

    train_dl = DataLoader(train_data, batch_size = params['batch_size'], shuffle=True, collate_fn=collate, pin_memory=True)
    test_dl = DataLoader(test_data, batch_size = params['batch_size'], shuffle=False, collate_fn=collate, pin_memory=True)
    logging.info("Data loaded successfully!")

    device = get_device(params["check_device"])
    model = BaseModel(params['nodes'], device, normal_avg, lr = params["learning_rate"], **params)


    # #For training!! 
    print("hash_id: ", hash_id)
    scores, converge = model.fit(train_dl, test_dl, evaluation_epoch= params['evaluation_epoch'])
    dump_scores(params["model_save_dir"], hash_id, scores, converge)
    logging.info("Current hash_id {}".format(hash_id))

    #For testing!!
    # model.load_model("./results/d07fe124/model.ckpt")
    # eval_result = model.evaluate(test_dl, datatype="Test")
    # eval_result = model.evaluate(test_dl, is_random=True, datatype="Test") #testing random ranked list 

def normal_status_average(train_data):
    result = {}
    sum_latency = None 
    sum_cpu = None
    sum_network_out = None 
    sum_network_in = None
    sum_memory = None

    normal_count = 0

    for data in train_data:
        if data[1] == 0: #anomaly가 없는 경우 normal 상태의 graph의 평균값을 얻기 위함
            normal_count += 1
            graph = data[0]
            if sum_latency is None:
                sum_latency = graph.ndata['latency']
                sum_cpu = graph.ndata['container_cpu_usage_seconds_total']
                sum_network_out = graph.ndata['container_network_transmit_bytes_total']
                sum_network_in = graph.ndata['container_network_receive_bytes_total']
                sum_memory = graph.ndata['container_memory_usage_bytes']
            else:
                sum_latency += graph.ndata['latency']
                sum_cpu += graph.ndata['container_cpu_usage_seconds_total']
                sum_network_out += graph.ndata['container_network_transmit_bytes_total']
                sum_network_in += graph.ndata['container_network_receive_bytes_total']
                sum_memory += graph.ndata['container_memory_usage_bytes']

    result['avg_latency'] = sum_latency / normal_count
    result['avg_cpu'] = sum_cpu / normal_count
    result['avg_network_out'] = sum_network_out / normal_count
    result['avg_network_in'] = sum_network_in / normal_count
    result['avg_memory'] = sum_memory / normal_count

    return result 

def extract_normal_status(train_data):
    result = {}

    for data in train_data:
        if data[1] == 0: #anomaly가 없는 경우 normal 상태의 graph의 평균값을 얻기 위함
            graph = data[0]
            result['avg_latency'] = graph.ndata['latency']
            result['avg_cpu'] = graph.ndata['container_cpu_usage_seconds_total']
            result['avg_network_out'] = graph.ndata['container_network_transmit_bytes_total']
            result['avg_network_in'] = graph.ndata['container_network_receive_bytes_total']
            result['avg_memory'] = graph.ndata['container_memory_usage_bytes']

    return result 


# Instantiate your Dataset and DataLoader
############################################################################
if __name__ == "__main__":

    nodes = 30
    batch_size = 256
    random_seed = 12345
    epochs = 30
    learning_rate = 0.001 
    model = "all"
    result_dir = "./results"
    each_modality_feature_num = 1 # 각각의 modality(latency, cpu ,, )마다 feature의 개수, 여기서는 모두 1
    chunk_length = 100 # window size 
    evaluation_epoch = 3

    train_file = "/root/projects/gamma/data/train_data_100_20.pkl"
    test_file = "/root/projects/gamma/data/test_data_100_20.pkl"

    call_path_file = "/root/projects/gamma/metadata/graph_path.txt"
    placement_file = "/root/projects/gamma/metadata/compose_ids.csv"

    features = ["latency", "container_cpu_usage_seconds_total", "container_memory_usage_bytes", 
                "container_network_transmit_bytes_total", "container_network_receive_bytes_total"]

    params = {'nodes': nodes,
            'batch_size': batch_size,
            'train_file': train_file,
            'test_file': test_file,
            'learning_rate': learning_rate, 
            'model': 'all',
            'check_device': "gpu",
            'input_dims': each_modality_feature_num,
            'model_save_dir': result_dir,
            'chunk_length' : chunk_length,
            'epochs' : epochs,
            'evaluation_epoch': evaluation_epoch,
            'call_path_file':call_path_file,
            'placement_file':placement_file
    }     

    run(params)



