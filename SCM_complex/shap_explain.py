import torch 
import torch.nn as nn
import numpy as np
from scm_simple_dataset import scm_dataset_gen, scm_out_of_domain, scm_diff_seed, scm_diff_model, scm_diff_rand_model, scm_indep_ood, scm_normal_dist, scm_indep
from scm_intv_simple_dataset import scm_intv_dataset_gen, scm_intv_ood, scm_intv_c_d_dataset_gen
from filename_funcs import get_filename, get_model_name
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
from scm_simple_network import MyDataset, DeepModel, TestModel

import sklearn
import shap

torch.manual_seed(2)
np.random.seed(2)

True_model = False

Scaling = False
Deep = True

Intervene = False 
C_D = False
Independent = True
Simplify = False

#Output_var = 'y1'
Output_var = 'y2'

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




#Make torch dataset - in this case, this is just to ensure the same scaling as when training the model
input_tensor = torch.from_numpy(inputs).float()
output_tensor = torch.from_numpy(outputs).float()

torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
# This implicitly shuffles the data as well, do don't need to include that in the dataset generator for the interventional data
train_data, val_data, test_data = torch.utils.data.random_split(torch_dataset, [int(0.7*n_datapoints), int(0.15*n_datapoints), int(0.15*n_datapoints)])  

if Scaling:
    trained_scaler_inputs = torch_dataset.fit_scaling_inputs(input_scaler, train_data)
    trained_scaler_outputs = torch_dataset.fit_scaling_outputs(output_scaler, train_data)



#Define a dataset with observational data, using a different seed from the training
n_obsv_testing = 500
obsv_inputs, obsv_targets = scm_dataset_gen(n_obsv_testing, seed = 54321)

input_tensor = torch.from_numpy(obsv_inputs).float()
output_tensor = torch.from_numpy(obsv_targets).float()

obsv_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    obsv_torch_dataset.scale_outputs(trained_scaler_outputs)



#Define a dataset with interventional data, using a different seed from the training
n_intv_testing = 500
intv_inputs, intv_targets = scm_intv_dataset_gen(n_intv_testing, seed = 54321)

input_tensor = torch.from_numpy(intv_inputs).float()
output_tensor = torch.from_numpy(intv_targets).float()

intv_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    intv_torch_dataset.scale_outputs(trained_scaler_outputs)




### Define an out-of-domain dataset with observational data
n_ood = 500

ood_inputs, ood_targets = scm_out_of_domain(n_ood)
input_tensor = torch.from_numpy(ood_inputs).float()
output_tensor = torch.from_numpy(ood_targets).float()
ood_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    ood_torch_dataset.scale_outputs(trained_scaler_outputs)

### Define an out-of-domain dataset with interventional data
ood_intv_inputs, ood_intv_targets = scm_intv_ood(n_ood)
input_tensor = torch.from_numpy(ood_intv_inputs).float()
output_tensor = torch.from_numpy(ood_intv_targets).float()
ood_intv_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    ood_intv_torch_dataset.scale_outputs(trained_scaler_outputs)



### Define a dataset with different SCM between the inputs
n_diff_model = 500
diff_mod_inputs, diff_mod_targets = scm_diff_model(n_diff_model)

input_tensor = torch.from_numpy(diff_mod_inputs).float()
output_tensor = torch.from_numpy(diff_mod_targets).float()

diff_mod_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    diff_mod_torch_dataset.scale_outputs(trained_scaler_outputs)


### Define a dataset with different SCM between the inputs, here the inputs are independent of each other
n_diff_rand_model = 500
diff_rand_mod_inputs, diff_rand_mod_targets = scm_diff_rand_model(n_diff_rand_model)

input_tensor = torch.from_numpy(diff_rand_mod_inputs).float()
output_tensor = torch.from_numpy(diff_rand_mod_targets).float()

diff_mod_rand_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    diff_mod_rand_torch_dataset.scale_outputs(trained_scaler_outputs)




# True data generating function, to compare 
def true_model(X):
    A = X[:, 0]
    D = X[:, 3]
    E = X[:, 4]

    y1 = 3.5*A + 0.5*D
    y2 = -2*D + 0.2*E

    if Output_var == 'y1':
        return y1
    else: 
        return y2




#f = lambda x: trained_model(torch.from_numpy(x)).detach().numpy()  #Wrap model to avoid problems with torch tensors and numpy arrays


#Wrap model to avoid problems with torch tensors and numpy arrays, plus used unscaled inputs and outputs for the Shap explainer
#Problem here is that the model and the trained_scalers must be global variables
def wrapping_func(x):
    if Scaling: 
        x = torch.from_numpy(trained_scaler_inputs.transform(x))
        x = x.to(torch.float32)
        pred = trained_model(x).detach().numpy()
        rescaled_pred = trained_scaler_outputs.inverse_transform(pred)
        return rescaled_pred
    
    else:
        pred = trained_model(torch.from_numpy(x)).detach().numpy()

        return pred


def shap_explainer(dataset, sample_size, n_samples):
    inputs, _ = dataset[:]
    inputs = inputs.numpy()
    # print(inputs.type())
    # print(inputs.shape)
    # exit()

    explainer = shap.KernelExplainer(wrapping_func, inputs[:sample_size, :])
    shap_values = explainer.shap_values(inputs[0:sample_size, :], nsamples=n_samples).squeeze()
    coefficients = np.divide(shap_values,(inputs[:sample_size, :]-np.mean(inputs[:sample_size, :], axis = 0)))

    avg_coeff = np.mean(coefficients, axis = 0)
    variance = np.var(coefficients, axis = 0)

    return avg_coeff, variance, coefficients


#Datasets available: obsv_torch_dataset, intv_torch_dataset, ood_torch_dataset, ood_intv_torch_dataset, diff_mod_torch_dataset, diff_mod_rand_torch_dataset

sample_size = 100
n_samples = 300


model_name = get_model_name(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
trained_model = torch.load(model_name)
trained_model.eval()

#trained_model = true_model

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

if True_model:
    filename = "true_model"
    trained_model = true_model

save_file = f"shap/{Output_var}/{filename}.csv"


file = open(save_file, "w")
file.write("Input_data,A,A_var,B,B_var,C,C_var,D,D_var,E,E_var,Avg_variances\n")
file.close()



dataset = obsv_torch_dataset 
avg_coeff, variance, coefficients = shap_explainer(dataset, sample_size, n_samples)
combined_coefficients = coefficients
combined_avg = np.expand_dims(avg_coeff, axis=0)
combined_variance = np.expand_dims(variance, axis=0)
file = open(save_file, "a")
avg_coeff, variance, coefficients = shap_explainer(dataset, sample_size, n_samples)
file.write(f"obsv,{avg_coeff[0]},{variance[0]},{avg_coeff[1]},{variance[1]},{avg_coeff[2]},{variance[2]},{avg_coeff[3]},{variance[3]},{avg_coeff[4]},{variance[4]},{np.mean(variance)}\n")
file.close()      


dataset = intv_torch_dataset 
avg_coeff, variance, coefficients = shap_explainer(dataset, sample_size, n_samples)
combined_coefficients = np.append(combined_coefficients, coefficients, axis = 0)
combined_avg = np.append(combined_avg, np.expand_dims(avg_coeff, axis=0), axis = 0)
combined_variance = np.append(combined_variance, np.expand_dims(variance, axis=0), axis = 0)
file = open(save_file, "a")
file.write(f"intv,{avg_coeff[0]},{variance[0]},{avg_coeff[1]},{variance[1]},{avg_coeff[2]},{variance[2]},{avg_coeff[3]},{variance[3]},{avg_coeff[4]},{variance[4]},{np.mean(variance)}\n")
file.close()     



dataset = ood_torch_dataset 
avg_coeff, variance, coefficients = shap_explainer(dataset, sample_size, n_samples)
combined_coefficients = np.append(combined_coefficients, coefficients, axis = 0)
combined_avg = np.append(combined_avg, np.expand_dims(avg_coeff, axis=0), axis = 0)
combined_variance = np.append(combined_variance, np.expand_dims(variance, axis=0), axis = 0)
file = open(save_file, "a")
file.write(f"ood_obsv,{avg_coeff[0]},{variance[0]},{avg_coeff[1]},{variance[1]},{avg_coeff[2]},{variance[2]},{avg_coeff[3]},{variance[3]},{avg_coeff[4]},{variance[4]},{np.mean(variance)}\n")
file.close()     

dataset = ood_intv_torch_dataset 
avg_coeff, variance, coefficients = shap_explainer(dataset, sample_size, n_samples)
combined_coefficients = np.append(combined_coefficients, coefficients, axis = 0)
combined_avg = np.append(combined_avg, np.expand_dims(avg_coeff, axis=0), axis = 0)
combined_variance = np.append(combined_variance, np.expand_dims(variance, axis=0), axis = 0)
file = open(save_file, "a")
file.write(f"ood_intv,{avg_coeff[0]},{variance[0]},{avg_coeff[1]},{variance[1]},{avg_coeff[2]},{variance[2]},{avg_coeff[3]},{variance[3]},{avg_coeff[4]},{variance[4]},{np.mean(variance)}\n")
file.close()     

dataset = diff_mod_torch_dataset 
avg_coeff, variance, coefficients = shap_explainer(dataset, sample_size, n_samples)
combined_coefficients = np.append(combined_coefficients, coefficients, axis = 0)
combined_avg = np.append(combined_avg, np.expand_dims(avg_coeff, axis=0), axis = 0)
combined_variance = np.append(combined_variance, np.expand_dims(variance, axis=0), axis = 0)
file = open(save_file, "a")
file.write(f"diff_mod,{avg_coeff[0]},{variance[0]},{avg_coeff[1]},{variance[1]},{avg_coeff[2]},{variance[2]},{avg_coeff[3]},{variance[3]},{avg_coeff[4]},{variance[4]},{np.mean(variance)}\n")
file.close()     

dataset = diff_mod_rand_torch_dataset 
avg_coeff, variance, coefficients = shap_explainer(dataset, sample_size, n_samples)
combined_coefficients = np.append(combined_coefficients, coefficients, axis = 0)
combined_avg = np.append(combined_avg, np.expand_dims(avg_coeff, axis=0), axis = 0)
combined_variance = np.append(combined_variance, np.expand_dims(variance, axis=0), axis = 0)
file = open(save_file, "a")
file.write(f"rand_mod,{avg_coeff[0]},{variance[0]},{avg_coeff[1]},{variance[1]},{avg_coeff[2]},{variance[2]},{avg_coeff[3]},{variance[3]},{avg_coeff[4]},{variance[4]},{np.mean(variance)}\n")
file.close()     

avg_coeff = np.mean(combined_coefficients, axis = 0)
variance = np.var(combined_coefficients, axis = 0)

file = open(save_file, "a")
file.write(f"Total,{avg_coeff[0]},{variance[0]},{avg_coeff[1]},{variance[1]},{avg_coeff[2]},{variance[2]},{avg_coeff[3]},{variance[3]},{avg_coeff[4]},{variance[4]},{np.mean(variance)}\n")
file.close()     

avg_combined_avg = np.mean(combined_avg, axis = 0)
#avg_combined_variance = np.mean(combined_variance, axis = 0)
avg_combined_variance = np.var(combined_avg, axis = 0)


file = open(save_file, "a")
file.write(f"Var_of_avg,{avg_combined_avg[0]},{avg_combined_variance[0]},{avg_combined_avg[1]},{avg_combined_variance[1]},{avg_combined_avg[2]},{avg_combined_variance[2]},{avg_combined_avg[3]},{avg_combined_variance[3]},{avg_combined_avg[4]},{avg_combined_variance[4]}\n")
file.close()    



file = open(save_file, "a")
file.write(f"Average_total_variance,{np.mean(variance)}\n")
file.close()     


