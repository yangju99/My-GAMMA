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


class BaseModel(nn.Module):
    def __init__(self, node_num, device, normal_avg, lr=1e-3, patience=5, result_dir='./results', hash_id=None, **kwargs):
        super(BaseModel, self).__init__()
        
        self.epochs = kwargs['epochs']
        self.lr = lr
        self.device = device
        self.node_num = node_num
        self.model_save_dir = os.path.join(kwargs['model_save_dir'], hash_id) 
        self.patience = patience 
        self.normal_avg = normal_avg

        if kwargs['model'] == 'all':
            self.model = MainModel(self.node_num, self.device, self.normal_avg, alpha=0.2, **kwargs)

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
    

    def evaluate(self, test_loader, is_random=False, datatype ="Test"):
        self.model.eval()
        batch_cnt, epoch_loss = 0, 0.0
        TP, FP, FN, TN = 0, 0, 0, 0 
        avg_my_metric = 0
        avg_avg_top = 0 
        A_1, A_3 = 0, 0 
        pred_type = "y_pred"

        ##### for prelimanary experiment ###################
        rootcause_avg_diff = 0
        normal_avg_diff = 0
        #####################################################


        if is_random:
            pred_type = "random_y_pred"
            
        with torch.no_grad():
            for graph, anomaly_gt, rootcause_gt in tqdm(test_loader, desc = "Testing progress"): #rootcause_gt = batchsize *[1,0,1,0,0 ...] len(rootcause_gt) = node_num 

                res = self.model.forward(graph.to(self.device), anomaly_gt, rootcause_gt)

                for idx, faulty_nodes in enumerate(res[pred_type]): #res['y_pred']는 anomaly 가 없다면 -1, 있다고 판단하면 RC node index list 를 값으로 가짐  
                    is_anomaly = anomaly_gt[idx].item()
                    if is_anomaly == 0:
                        if faulty_nodes[0] == -1: TN+=1
                        else: FP += 1
                    else:
                        if faulty_nodes[0] == -1: FN+=1
                        else: 
                            TP+=1  # 어떻게 RC이 여러개일때 RCL task에 대한 평가 metric을 정의할 것인가?
                            ranked_list = faulty_nodes
                            rootcause_indices = [i for i, value in enumerate(rootcause_gt[idx]) if value == 1]

                            # for Avg-top
                            rank_sum = 0  
                            for rc_index in rootcause_indices:
                                rank_sum += (ranked_list.index(rc_index) + 1) 
                            avg_top = rank_sum / len(rootcause_indices)
                            avg_avg_top += avg_top 
                              
                            k = len(rootcause_indices)
                            top_k_ranked_list = ranked_list[:k]

                            count = 0 
                            for index in top_k_ranked_list:
                                if index in rootcause_indices:
                                    count += 1

                            my_metric = count / k  
                            avg_my_metric += my_metric

                            top_1_ranked_list = ranked_list[:1]
                            for index in top_1_ranked_list:
                                if index in rootcause_indices:
                                    A_1 += 1
                            top_3_ranked_list = ranked_list[:3]
                            for index in top_3_ranked_list:
                                if index in rootcause_indices:
                                    A_3 += 1
                                    break

                epoch_loss += res["loss"].item()
                batch_cnt += 1
                if 'rootcause_avg_diff' in res.keys():
                    ##### for prelimanary experiment ###################
                    rootcause_avg_diff += res['rootcause_avg_diff'].item()
                    normal_avg_diff += res['normal_avg_diff'].item() 
                    #####################################################

            epoch_loss = epoch_loss / batch_cnt
            rootcause_avg_diff = rootcause_avg_diff / batch_cnt 
            normal_avg_diff = normal_avg_diff / batch_cnt 



        pos = TP+FN
        recall = TP*1.0/pos if pos > 0 else 0
        precision = TP*1.0/(TP+FP) if (TP+FP) > 0 else 0
        f1 = (2.0*recall*precision)/(precision+recall) if (precision+recall) > 0 else 0
        avg_my_metric = avg_my_metric / TP
        avg_avg_top = avg_avg_top / TP 
        A_1 = A_1 / TP
        A_3 = A_3 / TP 

        eval_results = {
                "loss": epoch_loss,
                "F1": f1,
                "Rec": recall,
                "Pre": precision,
                "My_metric":avg_my_metric,
                "Avg_top":avg_avg_top,
                "A@1": A_1,
                "A@3": A_3,
                "rootcause_avg_diff": rootcause_avg_diff,
                "normal_avg_diff": normal_avg_diff} 

        logging.info("{} -- {}".format(datatype, ", ".join([k+": "+str(f"{v:.4f}") for k, v in eval_results.items()])))

        return eval_results

    def fit(self, train_loader, test_loader=None, evaluation_epoch=1):
    ## initializing the fit function

        best_value, coverage, best_state, eval_res = -1, None, None, None # evaluation
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
                if test_results["F1"] > best_value:
                    best_value, eval_res, coverage  = test_results["F1"], test_results, epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                self.save_model(best_state)

        if coverage > 5:
            logging.info("* Best result got at epoch {} with F1: {:.4f}".format(coverage, best_value))
        else:
            logging.info("Unable to convergence!")

        return eval_res, coverage 

    

