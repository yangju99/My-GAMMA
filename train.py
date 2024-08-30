from torch.utils.data import Dataset, DataLoader
import torch
import dgl
from utils import * #collate, get_device 등의 함수 import 
import pickle
import sys
import logging
from base import BaseModel
import time

import pdb 

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class chunkDataset(Dataset): #[node_num, T]
    def __init__(self, chunks, edges, num_node):
        self.data = []
        self.idx2id = {}
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk["window_id"]
            self.idx2id[idx] = chunk_id
            graph = dgl.graph(edges)
            
            # (node_num, window_size) -> (window_size, node_num)
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

def run(params):

    train_file = params["train_file"]
    with open(train_file, 'rb') as tr:
        train_chunks = pickle.load(tr)
    tr.close()

    test_file = params["test_file"]
    with open(test_file, 'rb') as te:
        test_chunks = pickle.load(te)
    te.close()


    metadata = read_json(params["metadata_json"])
    source = [float(x) for x in metadata["source"]]
    target = [float(x) for x in metadata["target"]]


    edges = (source, target)
    train_data = chunkDataset(train_chunks, edges, params['nodes'])
    test_data = chunkDataset(test_chunks, edges, params['nodes'])
    
    pdb.set_trace() 

    train_dl = DataLoader(train_data, batch_size = params['batch_size'], shuffle=True, collate_fn=collate, pin_memory=True)
    test_dl = DataLoader(test_data, batch_size = params['batch_size'], shuffle=False, collate_fn=collate, pin_memory=True)
    logging.info("Data loaded successfully!")

    device = get_device(params["check_device"])
    model = BaseModel(nodes, device, epoches = evaluation_epochs, lr = params["learning_rate"], **params)
    train_logit_list = model.fit(train_dl)
    model = BaseModel(nodes, device, epoches = evaluation_epochs, lr = params["learning_rate"], **params)
    test_logit_list = model.evaluate(test_dl)  



# Instantiate your Dataset and DataLoader
############################################################################
if __name__ == "__main__":

    nodes = 30
    batch_size = 32
    random_seed = 12345
    evaluation_epochs = 20
    learning_rate = 0.001
    model = "all"
    result_dir = "./results"
    window_size = 60 # 여기서의 window_size는 time-temporal 을 위한 1D convolution layer의 output_dim 을 뜻함? 

    train_file = "../data/train_data.pkl"
    test_file = "../data/test_data.pkl"
    metadata_json = "../data/metadata.json"

    features = ["latency", "container_cpu_usage_seconds_total", "container_memory_usage_bytes", 
                "container_network_transmit_bytes_total", "container_network_receive_bytes_total"]


    params = {'nodes': nodes,
            'batch_size': batch_size,
            'train_file': train_file,
            'test_file': test_file,
            'metadata_json': metadata_json,
            'learning_rate': learning_rate, 
            'model': 'all',
            'check_device': "gpu",
            'input_dims': window_size,
            'model_save_dir': result_dir
    }     

    run(params)
    # print(device)
    # print(type(device))


