from pysr import PySRRegressor
import torch
import torch.nn as nn
import numpy as np
from scm_complex_dataset import mixed_dataset_gen
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


#Output_var = 'y1'
Output_var = 'y2'

n_testing = 600
no_ood = False

Simplify = False
eq_path = f"pysr/{Output_var}/true_model/" 

inputs, targets = mixed_dataset_gen(n_testing, Output_var, seed = 54321)
input_tensor = torch.from_numpy(inputs).float()
output_tensor = torch.from_numpy(targets).float()
torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
unscaled_inputs, unscaled_targets = torch_dataset[:]


datasets = ["obsv", "intv", "ood", "ood_intv", "diff_mod", "diff_mod_rand"]
model_save_filename = f"pysr/{Output_var}/true_model/summary.csv"

if no_ood:
    datasets = ["obsv", "intv", "diff_mod", "diff_mod_rand"]
    model_save_filename = f"pysr/{Output_var}/true_model/summary_no_ood.csv"

model_save_file = open(model_save_filename, "w")
model_save_file.write("Input_data,loss,variance\n") 

all_preds = np.zeros((len(datasets), n_testing))

nan_idx_list = []
all_losses = []

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

    all_losses.append(loss)
    model_save_file.write(f"{data},{loss}\n") 

variance = np.var(all_preds, axis = 0)
avg_variance = np.mean(variance)
avg_loss = np.mean(all_losses)
model_save_file.write(f"Average,{avg_variance},{avg_loss}\n") 
