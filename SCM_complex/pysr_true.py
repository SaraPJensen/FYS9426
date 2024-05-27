from pysr import PySRRegressor
import torch
import torch.nn as nn
import numpy as np
from scm_complex_dataset import scm_dataset_gen_inclusive, scm_dataset_gen, scm_out_of_domain, scm_diff_seed, scm_diff_model, scm_diff_rand_model, scm_indep_ood, scm_normal_dist, scm_indep
from scm_intv_complex_dataset import scm_intv_dataset_gen, scm_intv_ood, scm_intv_c_d_dataset_gen
from filename_funcs import get_filename, get_model_name
from math import sqrt, exp, sin, cos
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
from scm_complex_network import MyDataset, DeepModel, TestModel
import sklearn
import pandas as pd

np.random.seed(2)


Output_var = 'y1'
#Output_var = 'y2'

n_testing = 500

Simplify = False
eq_path = f"pysr/{Output_var}/true_model/" 

inputs, targets = scm_dataset_gen_inclusive(n_testing, seed = 54321)
input_tensor = torch.from_numpy(inputs).float()
output_tensor = torch.from_numpy(targets).float()
torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
unscaled_inputs, unscaled_targets = torch_dataset[:]


datasets = ["obsv", "intv", "ood", "ood_intv", "diff_mod", "diff_mod_rand"]


all_preds = np.zeros((len(datasets), n_testing))

model_save_filename = f"pysr/{Output_var}/true_model/summary.csv"
model_save_file = open(model_save_filename, "w")
model_save_file.write("Input_data,loss,variance\n") 

nan_idx_list = []

for count, data in enumerate(datasets):
    model_file = eq_path + data + ".pkl"

    retrieved_mod = PySRRegressor.from_file(model_file)

    preds = retrieved_mod.predict(unscaled_inputs.numpy())

    all_preds[count] = preds

    if np.isnan(preds).sum() > 0:  #Turns out some of the elements in the predictions are NaN. Remove these from the datasets
        nan_idx = np.sort(np.where(np.isnan(preds)))
        new_preds = np.delete(preds, nan_idx)
        new_targets = np.delete(unscaled_targets.numpy(), nan_idx)
        loss = sklearn.metrics.mean_squared_error(new_targets, new_preds)

        nan_idx_list.append(nan_idx)
    
    else: 
        loss = sklearn.metrics.mean_squared_error(unscaled_targets.numpy(), preds)

    model_save_file.write(f"{data},{loss}\n") 

variance = np.var(all_preds, axis = 0)
avg_variance = np.mean(variance)
model_save_file.write(f",,{avg_variance}\n") 
