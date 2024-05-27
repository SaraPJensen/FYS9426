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


def pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_save_filename, Output_var, Deep, Scaling, Intv, C_D, Independent):
    all_preds = np.zeros((len(datasets), n_testing))
    all_losses = np.zeros((len(datasets),))

    model_save_filename = f"pysr/{Output_var}/{filename}/summary.csv"
    model_save_file = open(model_save_filename, "w")
    model_save_file.write("Input_data,loss\n") 

    nan_idx_list = []

    for count, data in enumerate(datasets):
        model_file = eq_path + data + ".pkl"

        retrieved_mod = PySRRegressor.from_file(model_file)

        preds = retrieved_mod.predict(unscaled_inputs.numpy())
        # print(preds)

        # print()
        all_preds[count] = preds

        if np.isnan(preds).sum() > 0:  #Turns out some of the elements in the predictions are NaN. Remove these from the datasets
            nan_idx = np.sort(np.where(np.isnan(preds)))
            new_preds = np.delete(preds, nan_idx)
            new_targets = np.delete(unscaled_targets.numpy(), nan_idx)
            loss = sklearn.metrics.mean_squared_error(new_targets, new_preds)

            nan_idx_list.append(nan_idx)
        
        else: 
            loss = sklearn.metrics.mean_squared_error(unscaled_targets.numpy(), preds)

        all_losses[count] = loss
        model_save_file.write(f"{data},{loss}\n") 

    if len(nan_idx_list):
        if len(nan_idx_list) > 1:
            print(len(nan_idx_list))  #If there are more elements, the current method doesn't work, but luckily there isn't
            exit()

        all_preds = np.delete(all_preds, nan_idx_list, axis = 1)


    variance = np.var(all_preds, axis = 0)

    avg_variance = np.mean(variance) 
    avg_loss = np.mean(all_losses)

    if Scaling: 
        Scale_type = "MinMax"
    else:
        Scale_type = "Raw_data"

    save_file = open(summary_save_filename, "a")
    save_file.write(f"{Deep},{Scale_type},{Intv},{C_D},{Independent},{avg_variance},{avg_loss}\n")  
    save_file.close()








True_model = False

Scaling = False
Deep = False

Intervene = False 
C_D = False
Independent = False
Simplify = False

#Output_var = 'y1'
Output_var = 'y2'

no_ood = True

#Define a dataset with observational data with an extended range
n_testing = 500

inputs, targets = scm_dataset_gen_inclusive(n_testing, seed = 54321)
input_tensor = torch.from_numpy(inputs).float()
output_tensor = torch.from_numpy(targets).float()
torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
unscaled_inputs, unscaled_targets = torch_dataset[:]


datasets = ["obsv", "intv", "ood", "ood_intv", "diff_mod", "diff_mod_rand"]

if no_ood:
    datasets = ["obsv", "intv", "diff_mod", "diff_mod_rand"]
    summary_filename = f"pysr/{Output_var}/{Output_var}_no_ood_summary.csv"
else:
    summary_filename = f"pysr/{Output_var}/{Output_var}_summary.csv"

summary_file = open(summary_filename, "w")
summary_file.write("Deep,Scaling,Intv,C_D,Independent,Avg_variance,Avg_loss\n")
summary_file.close()




filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)




Intervene = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)




C_D = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



Intervene = False
C_D = False
Independent = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)





Deep = False
Scaling = True

Intervene = False
C_D = False
Independent = False
Simplify = False


filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



Intervene = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



C_D = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



Intervene = False
C_D = False
Independent = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)





Deep = True
Scaling = False

Intervene = False
C_D = False
Independent = False
Simplify = False


filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



Intervene = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



C_D = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



Intervene = False
C_D = False
Independent = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)




Deep = True
Scaling = True

Intervene = False
C_D = False
Independent = False
Simplify = False


filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



Intervene = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



C_D = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



Intervene = False
C_D = False
Independent = True

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
eq_path = f"pysr/{Output_var}/{filename}/" 
pysr_summary(datasets, eq_path, unscaled_inputs, unscaled_targets, n_testing, filename, summary_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)











