#!/usr/bin/env python
# coding: utf-8
from numpy.lib.npyio import save
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import  load_model
from hd_gcn import GCNLayer, GCNPooling
from hd_utils import load_data, save_txt, norm_test, all_permutation, res_with_label, res_without_label
from time import localtime, strftime, time, perf_counter
import os

if __name__ == "__main__":

    dst_dir = 'D:/RecordGcn/res_cnn'+strftime('%m%d_%H%M%S', localtime(time()))
    strlog = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
    strlog += strftime('predict: %Y-%m-%d: %H:%M:%S:', localtime(time()))
    save_txt(file=dst_dir+'/trainRecord.txt', fmode='a+', str=strlog)
  
    feature_file = "./data/feature_0X16_RS232-T2000_iCTRL.txt"
    edge_file = "./data/edge_0X16_RS232-T2000_iCTRL.txt"
    data_res = load_data(feature_file=feature_file,edge_file=edge_file, train=0)    

    if len(data_res)==5:
        x, adj, y, edge_name, gate_label = data_res
    elif len(data_res)==3:
        x, adj, edge_name = data_res
    else:
        strlog = data_res
        print(strlog)
        save_txt(file=dst_dir+'/trainRecord.txt', fmode='a+', str=strlog)
        exit()
    

    adj_graph = adj
    
    ## normalization
    x, adj = norm_test(x, adj)
 
    model_input_test = [x, adj]
    feature_dim = x.shape[-1]

    #load model
    model = load_model('./savedModel/best_model_pred.h5', custom_objects={'GCNLayer':GCNLayer, 'GCNPooling':GCNPooling})
    model.summary()

    match_list = np.arange(x.shape[0]).tolist()
    if len(data_res)==5:
        eval_results = model.evaluate(model_input_test, y, batch_size=adj.shape[0])
        val_res = ('Trainging Done.\nTest loss: {}\nTest weighted_loss: {}\nTest accuracy: {}'.format(*eval_results))
        print(val_res)
        save_txt(file=dst_dir+'/trainRecord.txt', fmode='a+',  str=val_res)
        
        start = perf_counter()

        y_predict = model.predict(model_input_test, batch_size=adj.shape[0])   #predict time

        end = perf_counter()
        
        str_predict_run_time = (f'predict time: {end-start}s')
        save_txt(file=dst_dir+'/trainRecord.txt', fmode='a+',  str=str_predict_run_time)
        print(str_predict_run_time)
        
        res = res_with_label(y_predict, y, match_list, 0, edge_name, gate_label, x, adj_graph, y, dst_dir=dst_dir)    
    elif len(data_res)==3:

        y_predict = model.predict(model_input_test, batch_size=adj.shape[0])
        res = res_without_label(y_predict, match_list, 0, edge_name, x, adj_graph, dst_dir=dst_dir)

    
    print(res)
    save_txt(file=dst_dir+'/trainRecord.txt', fmode='a+', str=res)  



