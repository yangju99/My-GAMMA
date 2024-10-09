import os
import logging
import pickle
import torch
import numpy as np
import random
import json
import logging
import dgl
import inspect
import pdb 

def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else: 
        logging.raiseExceptions("File path "+filepath+" not exists!")
        return

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(gpu):
    if gpu and torch.cuda.is_available():
        logging.info("Using GPU...")
        return torch.device("cuda")
    logging.info("Using CPU...")
    return torch.device("cpu")


def get_metrics(predictions, true_labels):
    assert len(predictions) == len(true_labels), "Length mismatch between predictions and true_labels."

    TP, TN, FP, FN = 0, 0, 0, 0
    
    for pred, true in zip(predictions, true_labels):
        if pred == 1 and true == 1:
            TP += 1
        elif pred == 0 and true == 0:
            TN += 1
        elif pred == 1 and true == 0:
            FP += 1
        elif pred == 0 and true == 1:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return accuracy, precision, recall, f1_score


def collate(data):
    graphs, anomaly_gts, rootcause_gts = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    anomaly_gts_array = np.array(anomaly_gts)
    rootcause_gts_array = np.array(rootcause_gts)

    return batched_graph , torch.tensor(anomaly_gts_array), torch.tensor(rootcause_gts_array)


def save_logits_as_dict(logits, keys, filename):
    """
    Saves a list of tensors as a dictionary with variable names as keys and tensor values as dictionary values.
    """
    # Get the previous frame (caller frame)
    frame = inspect.currentframe().f_back
    tensor_dict = {}
    
    # Loop through the tensors
    for logit in logits:
        # Find all variable names that this tensor object is assigned to
        names = [name for name, var in frame.f_locals.items() if torch.is_tensor(var) and var is tensor and not name.startswith('_')]
        
        # If there's a name that refers to this tensor, add to dictionary
        if names:
            tensor_dict[names[0]] = logit
            
    return tensor_dict

import hashlib
def dump_params(params):
    hash_id = hashlib.md5(str(sorted([(k, v) for k, v in params.items()])).encode("utf-8")).hexdigest()[0:8]
    result_dir = os.path.join(params["model_save_dir"], hash_id)
    os.makedirs(result_dir, exist_ok=True)

    json_pretty_dump(params, os.path.join(result_dir, "params.json"))

    log_file = os.path.join(result_dir, "running.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return hash_id

from datetime import datetime, timedelta
def dump_scores(result_dir, hash_id, scores, converge):
    with open(os.path.join(result_dir, 'experiments.txt'), 'a+') as fw:
        fw.write(hash_id+': '+(datetime.now()+timedelta(hours=8)).strftime("%Y/%m/%d-%H:%M:%S")+'\n')
        fw.write("* Test result -- " + '\t'.join(["{}:{:.4f}".format(k, v) for k,v in scores.items()])+'\n')
        fw.write('Best score got at epoch: '+str(converge)+'\n')
        fw.write('{}{}'.format('='*40, '\n'))


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj,fw, sort_keys=True, indent=4, separators=(",", ": "), ensure_ascii=False)


# def anomaly_score_check(y_prob, score_diff_list):
    
#     pdb.set_trace() 
#     normal_score = 0 
#     normal_count = 0 
#     rootcause_score = 0 
#     rootcause_count = 0

#     for i in range(len(score_diff_list)):
#         if score_diff_list[i] == -1:
#             continue 

#         rc_gt = y_prob[i]
#         diff_list = score_diff_list[i] 
#         # TP 일때만 계산 
#         if 1 in rc_gt: 
#             for j in range(len(rc_gt)):
#                 if rc_gt[j] == 0:
#                     normal_score += diff_list[j]
#                     normal_count += 1

#                 else:
#                     rootcause_score += diff_list[j]
#                     rootcause_count += 1
                
        
#     return normal_score/normal_count, rootcause_score/rootcause_count 

