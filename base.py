import os
import time
import copy
import sys
import torch
from torch import nn
import logging
from utils import *
from models import MainModel
import pdb 
from tqdm import tqdm 
# from models_cpu import EvaluateCpu
# from models_memory import EvaluateMemory
# from models_network import EvaluateNetwork
# from models_mixed import EvaluateMixed

class BaseModel(nn.Module):
    def __init__(self, node_num, device, lr=1e-3, patience=5, result_dir='./results', hash_id=None, **kwargs):
        super(BaseModel, self).__init__()
        
        self.epochs = kwargs['epochs']
        self.lr = lr
        self.device = device
        self.node_num = node_num
        self.model_save_dir = os.path.join(kwargs['model_save_dir'], hash_id) 
        self.patience = patience 

        # self.model_save_dir = os.path.join(result_dir, hash_id)
        if kwargs['model'] == 'all':
            self.model = MainModel(self.node_num, self.device, alpha=0.2, **kwargs)
        # elif kwargs['model'] == 'cpu':
        #     self.model = EvaluateCpu(self.node_num, self.device, alpha=0.2, **kwargs)
        # elif kwargs['memory'] == 'memory':
        #     self.model = EvaluateMemory(self.node_num, self.device, alpha=0.2, **kwargs)
        # elif kwargs['network'] == 'network':
        #     self.model = EvaluateNetwork(self.node_num, self.device, alpha=0.2, **kwargs)
        # elif kwargs['mixed'] == 'mixed':
        #     self.model = EvaluateMixed(self.node_num, self.device, alpha=0.2, **kwargs)
        else:
            print("Please select a valid model")
            sys.exit(1)
        self.model.to(device)


    def save_model(self, state, file=None):
        if file is None: file = os.path.join(self.model_save_dir, "model.ckpt")
        try:
            torch.save(state, file, _use_new_zipfile_serialization=False)
        except:
            torch.save(state, file)
    

    def load_model(self, model_save_file=""):
        self.model.load_state_dict(torch.load(model_save_file, map_location=self.device))
    

    def evaluate(self, test_loader, datatype ="Test"):
        self.model.eval()
        my_metric = [] # 전체 rc에서 몇퍼센트나 잡아내는가?
        random_my_metric = []
        batch_cnt, epoch_loss = 0, 0.0
        TP, FP, FN, TN = 0, 0, 0, 0 
        detect_percentage = 0.0
        detect_count = 0.0

        count_for_score_check = 0 
        avg_max_score_diff = 0.0
        avg_gt_score_diff = 0.0 

        with torch.no_grad():
            for graph, anomaly_gt, rootcause_gt in tqdm(test_loader, desc = "Testing progress"): #rootcause_gt = batchsize *[1,0,1,0,0 ...] len(rootcause_gt) = node_num 

                res = self.model.forward(graph.to(self.device), anomaly_gt, rootcause_gt)

                for k in range(len(res['y_pred'])):
                    if res['max_score_diff'][k] != -1 and res['gt_score_diff'][k] != -1:
                        avg_gt_score_diff += res['gt_score_diff'][k]
                        avg_max_score_diff += res['max_score_diff'][k]
                        count_for_score_check += 1
                        


                for idx, faulty_nodes in enumerate(res["y_pred"]): #res['y_pred']는 anomaly 가 없다면 -1, 있다고 판단하면 RC node index list 를 값으로 가짐  
                    is_anomaly = anomaly_gt[idx].item()
                    if is_anomaly == 0:
                        if faulty_nodes[0] == -1: TN+=1
                        else: FP += 1
                    else:
                        if faulty_nodes[0] == -1: FN+=1
                        else: 
                            TP+=1  # 어떻게 RC이 여러개일때 RCL task에 대한 평가 metric을 정의할 것인가?
                            gt_rootcause_indexs = [index for index, value in enumerate(rootcause_gt[idx]) if value == 1]
                            pred_rootcause_indexs = faulty_nodes

                            count = 0
                            for rc in gt_rootcause_indexs:
                                if rc in pred_rootcause_indexs:
                                    count+= 1
                            if count != 0:
                                detect_count += 1

                            detect_percentage += count / len(gt_rootcause_indexs)

                epoch_loss += res["loss"].item()
                batch_cnt += 1

            epoch_loss = epoch_loss / batch_cnt
            avg_max_score_diff = avg_max_score_diff / count_for_score_check
            avg_gt_score_diff = avg_gt_score_diff / count_for_score_check

        pos = TP+FN
        recall = TP*1.0/pos if pos > 0 else 0
        precision = TP*1.0/(TP+FP) if (TP+FP) > 0 else 0
        f1 = (2.0*recall*precision)/(precision+recall) if (precision+recall) > 0 else 0
        eval_results = {
                "loss": epoch_loss,
                "F1": f1,
                "Rec": recall,
                "Pre": precision,
                "Metric 1": detect_count / TP, #root cause를 하나라도 맞춘 비율 
                "Metric 2": detect_percentage / TP,
                "avg_max_score_diff" : avg_max_score_diff,
                "avg_gt_score_diff" : avg_gt_score_diff} #실제 root cause 중 pred 성공한 비율 평균  

        logging.info("{} -- {}".format(datatype, ", ".join([k+": "+str(f"{v:.4f}") for k, v in eval_results.items()])))

        return eval_results

    def fit(self, train_loader, test_loader=None, evaluation_epoch=1):
    ## initializing the fit function

        best_f1, coverage, best_state, eval_res = -1, None, None, None # evaluation
        pre_loss, worse_count = float("inf"), 0 # early break

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()

            for graph, anomaly_gt, rootcause_gt in tqdm(train_loader, desc="Training progress"): #rootcause_gt = [1,0,1,0,0 ...] len(rootcause_gt) = node_num 
                optimizer.zero_grad()
                res = self.model.forward(graph.to(self.device), anomaly_gt, rootcause_gt, only_train=True)
                loss = res['loss']
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt
            logging.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epochs, epoch_loss, epoch_time_elapsed))
    
            ####### early break #######
            if epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break
            else: 
                worse_count = 0
            pre_loss = epoch_loss

            ####### Evaluate test data during training #######
            if (epoch+1) % evaluation_epoch == 0:
                test_results = self.evaluate(test_loader, datatype="Test")
                if test_results["F1"] > best_f1:
                    best_f1, eval_res, coverage  = test_results["F1"], test_results, epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                self.save_model(best_state)

        if coverage > 5:
            logging.info("* Best result got at epoch {} with Mymetric: {:.4f}".format(coverage, best_f1))
        else:
            logging.info("Unable to convergence!")

        return eval_res, coverage 

    

