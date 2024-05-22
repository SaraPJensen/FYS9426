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

True_model = False

Scaling = True
Deep = True

Intervene = True 
C_D = False
Independent = False
Simplify = False

#Output_var = 'y1'
Output_var = 'y2'




#Make torch dataset - this is just for scaling
n_datapoints = 3000
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

if Intervene:
    if C_D:
        inputs, outputs = scm_intv_c_d_dataset_gen(n_datapoints)
    else:
        inputs, outputs = scm_intv_dataset_gen(n_datapoints)

elif Independent:
    inputs, outputs = scm_indep(n_datapoints)
else:
    inputs, outputs = scm_dataset_gen(n_datapoints)

input_tensor = torch.from_numpy(inputs).float()
output_tensor = torch.from_numpy(outputs).float()


scaling_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
# This implicitly shuffles the data as well, do don't need to include that in the dataset generator for the interventional data
train_data, val_data, test_data = torch.utils.data.random_split(scaling_torch_dataset, [int(0.7*n_datapoints), int(0.15*n_datapoints), int(0.15*n_datapoints)])  

if Scaling:
    trained_scaler_inputs = scaling_torch_dataset.fit_scaling_inputs(input_scaler, train_data)
    trained_scaler_outputs = scaling_torch_dataset.fit_scaling_outputs(output_scaler, train_data)







#Define a dataset with observational data, using a different seed from the training
n_testing = 5

inputs, targets = scm_dataset_gen_inclusive(n_testing, seed = 54321)

input_tensor = torch.from_numpy(inputs).float()
output_tensor = torch.from_numpy(targets).float()

torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
unscaled_inputs, unscaled_targets = torch_dataset[:]

if Scaling:
    torch_dataset.scale_inputs(trained_scaler_inputs)
    torch_dataset.scale_outputs(trained_scaler_outputs)

scaled_inputs, scaled_targets = torch_dataset[:]  #For the ML model



filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
save_path = f"pysr/{Output_var}/{filename}/" 
model_name = get_model_name(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
trained_model = torch.load(model_name)
trained_model.eval()

datasets = ["obsv", "intv", "ood", "ood_intv", "diff_mod", "diff_mod_rand"]

all_preds = np.zeros((6, n_testing))
all_losses = []

ml_to_pysr_losses = []

with torch.no_grad():
    ml_pred = trained_model(scaled_inputs)

rescaled_ml_pred = torch_dataset.rescale_outputs(ml_pred)


for count, data in enumerate(datasets):
    model_file = save_path + data + ".pkl"

    retrieved_mod = PySRRegressor.from_file(model_file)

    preds = retrieved_mod.predict(unscaled_inputs.numpy())
    all_preds[count] = preds

    loss = sklearn.metrics.mean_squared_error(unscaled_targets, preds)
    all_losses.append(loss)

    ml_to_pysr_losses.append(sklearn.metrics.mean_squared_error(rescaled_ml_pred, preds))


print()
print(all_preds)


print()


print("True values: ", targets)

print()
print(ml_pred)


print()


print("All losses: ", all_losses)

print("ML to pysr losses: ", ml_to_pysr_losses)


loss_torch = nn.MSELoss()


torch_ml_loss = loss_torch(scaled_targets, ml_pred)

unscaled_loss = loss_torch(unscaled_targets, rescaled_ml_pred)

#print("Torch ml loss, scaled: ", torch_ml_loss.item())
print("Torch ml loss, unscaled: ", unscaled_loss.item())








'''



all_funcs = pd.read_csv(save_file)

eqs = all_funcs['expression']





print(sqrt(5))


first_point = obsv_inputs[0:2,:]

first_point = [1, 2, 3, 4, 5]

print(first_point)

dict_to_eval = {"a": first_point[0], "b": first_point[1], "c": first_point[2], "d": first_point[3], "e": first_point[4], "sqrt": sqrt, "sin": sin, "cos": cos, "exp": exp}


test_ex_0 = eval(eqs[0], dict_to_eval)
print("First eq:", test_ex_0)

test_ex_1 = eval(eqs[1], dict_to_eval)
print("Second eq: ", test_ex_1)










all_funcs = pd.read_csv(save_file)

eqs = all_funcs['expression']



first_point = obsv_inputs[0,:].reshape(1, -1)
print(first_point)

print("Obsv true: ", obsv_model.predict(first_point))
print("Intv true: ", intv_model.predict(first_point))

#first_point = [1, 2, 3, 4, 5]



first_point = obsv_inputs[0,:]
dict_to_eval = {"a": first_point[0], "b": first_point[1], "c": first_point[2], "d": first_point[3], "e": first_point[4], "sqrt": sqrt, "sin": sin, "cos": cos, "exp": exp}


test_ex_0 = eval(eqs[0], dict_to_eval)
print("Obsv eq:", test_ex_0)


test_ex_1 = eval(eqs[1], dict_to_eval)
print("Intv eq: ", test_ex_1)





dataset_name = "obsv"

retrieve_mod = PySRRegressor.from_file(f"{save_path}/{dataset_name}.pkl")

first_point = obsv_inputs[0:3,:]

print(retrieve_mod.predict(first_point))
exit()

'''