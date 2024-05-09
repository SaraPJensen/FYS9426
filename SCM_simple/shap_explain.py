import torch 
import torch.nn as nn
import numpy as np
from SCM_simple.scm_simple_dataset import scm_dataset_gen, scm_out_of_domain, scm_diff_seed, Franke_data, super_simple, scm_diff_model, scm_diff_rand_model
from SCM_simple.scm_intv_simple_dataset import scm_intv_dataset_gen, scm_intv_diff_seed, scm_intv_ood, scm_intv_c_d_dataset_gen
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.init as init
from lime import lime_tabular
from sklearn.preprocessing import MinMaxScaler
from SCM_simple.scm_simple_network import MyDataset, TestModel



torch.manual_seed(2)
np.random.seed(2)


#regions Parameters
#Hyperparameters
learning_rate = 1e-3
epochs = 1000
Input_scaling = True
Output_scaling = True
Intervene = True
Intervene_info = False
C_D = True

Output_var = 'y1'
#Output_var = 'y2'



if __name__ == "__main__":

    #region Make dataset 

    n_datapoints = 500
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    if Intervene:
        if C_D:
            inputs, outputs = scm_intv_c_d_dataset_gen(n_datapoints, Intervene_info)
        else:
            inputs, outputs = scm_intv_dataset_gen(n_datapoints, Intervene_info)

    else:
        inputs, outputs = scm_dataset_gen(n_datapoints)


    batch_size = 32

    #Make torch dataset
    input_tensor = torch.from_numpy(inputs).float()
    output_tensor = torch.from_numpy(outputs).float()

    torch_dataset = MyDataset(input_tensor, output_tensor, Output_var) 
    # This implicitly shuffles the data as well, do don't need to include that in the dataset generator for the interventional data
    train_data, val_data, test_data = torch.utils.data.random_split(torch_dataset, [int(0.7*n_datapoints), int(0.15*n_datapoints), int(0.15*n_datapoints)])  

