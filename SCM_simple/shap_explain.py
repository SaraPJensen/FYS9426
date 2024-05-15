import torch 
import torch.nn as nn
import numpy as np
from scm_simple_dataset import scm_dataset_gen, scm_out_of_domain, scm_diff_seed, scm_diff_model, scm_diff_rand_model, scm_indep_ood, scm_normal_dist, scm_indep
from scm_intv_simple_dataset import scm_intv_dataset_gen, scm_intv_ood, scm_intv_c_d_dataset_gen
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.init as init
from lime import lime_tabular
from sklearn.preprocessing import MinMaxScaler
from scm_simple_network import MyDataset, DeepModel, TestModel
from torch.autograd import Variable

import sklearn
import shap

torch.manual_seed(2)
np.random.seed(2)

Scaling = True
Deep = False

Intervene = True 
C_D = False
Independent = False
Simplify = False

#Output_var = 'y1'
Output_var = 'y2'

n_datapoints = 3000
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()


def get_model_name(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify):

    file_intv = ''
    if Intervene:
        if C_D:
            file_intv = "Intv_C_D_"
        else:
            file_intv = "Intv_"

    if Scaling:
        filename = f"{Output_var}_{file_intv}MinMax_lr_0.001"
    else:
        filename = f"{Output_var}_{file_intv}Raw_data_lr_0.001"

    if Independent:
        filename = "indep_" + filename

    if Simplify: 
        filename = "simple_" + filename

    if Deep:
        filename = "deeeeep_" + filename

    model_name = f"saved_models/{Output_var}/best_{filename}.pth"

    return model_name


if Intervene:
    if C_D:
        inputs, outputs = scm_intv_c_d_dataset_gen(n_datapoints)
    else:
        inputs, outputs = scm_intv_dataset_gen(n_datapoints)

elif Independent:
    inputs, outputs = scm_indep(n_datapoints)
else:
    inputs, outputs = scm_dataset_gen(n_datapoints)


batch_size = 32

#Make torch dataset
input_tensor = torch.from_numpy(inputs).float()
output_tensor = torch.from_numpy(outputs).float()


torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
# This implicitly shuffles the data as well, do don't need to include that in the dataset generator for the interventional data
train_data, val_data, test_data = torch.utils.data.random_split(torch_dataset, [int(0.7*n_datapoints), int(0.15*n_datapoints), int(0.15*n_datapoints)])  

if Scaling:
    trained_scaler_inputs = torch_dataset.fit_scaling_inputs(input_scaler, train_data)
    trained_scaler_outputs = torch_dataset.fit_scaling_outputs(output_scaler, train_data)


### Define an out-of-domain dataset
n_ood = 500

if Intervene:
    ood_inputs, ood_targets = scm_intv_ood(n_ood)
elif Independent:
    ood_inputs, ood_targets = scm_indep_ood(n_ood)
else:
    ood_inputs, ood_targets = scm_out_of_domain(n_ood)

input_tensor = torch.from_numpy(ood_inputs).float()
output_tensor = torch.from_numpy(ood_targets).float()

ood_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    ood_torch_dataset.scale_outputs(trained_scaler_outputs)
#ood_test_loader = torch.utils.data.DataLoader(ood_torch_dataset, batch_size = 1, shuffle = True)


### Define a dataset with different SCM between the inputs
#For now, this is never interventional 

n_diff_model = 500
diff_mod_inputs, diff_mod_targets = scm_diff_model(n_diff_model)

input_tensor = torch.from_numpy(diff_mod_inputs).float()
output_tensor = torch.from_numpy(diff_mod_targets).float()

diff_mod_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    diff_mod_torch_dataset.scale_outputs(trained_scaler_outputs)
#diff_mod_loader = torch.utils.data.DataLoader(diff_mod_torch_dataset, batch_size = 1, shuffle = True)


### Define a dataset with different SCM between the inputs, here the inputs are independent of each other
#For now, this is never interventional 

n_diff_rand_model = 500
diff_rand_mod_inputs, diff_rand_mod_targets = scm_diff_rand_model(n_diff_rand_model)

input_tensor = torch.from_numpy(diff_rand_mod_inputs).float()
output_tensor = torch.from_numpy(diff_rand_mod_targets).float()

diff_mod_rand_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    diff_mod_rand_torch_dataset.scale_outputs(trained_scaler_outputs)
#diff_mod_rand_loader = torch.utils.data.DataLoader(diff_mod_rand_torch_dataset, batch_size = 1, shuffle = True)








def true_model(X):
    A = X[:,0]
    D = X[:, 3]
    E = X[:, 4]

    y1 = 3.5*A + 0.5*D
    y2 = -2*D + 0.2*E

    if Output_var == 'y1':
        return y1
    else: 
        return y2


feature_names = ['A', 'B', 'C', 'D', 'E']


test_input, test_output = test_data[:]
test_input = np.asarray(test_input.numpy())
test_output = np.asarray(test_output.numpy())

# model = sklearn.linear_model.LinearRegression()
# model.fit(test_input, test_output)

# print(model.coef_[0])


model_name = get_model_name(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
trained_model = torch.load(model_name)
trained_model.eval()



f = lambda x: trained_model(torch.from_numpy(x)).detach().numpy()  #Wrap model to avoid problems with torch tensors and numpy arrays


sample_size = 30

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




explainer = shap.KernelExplainer(wrapping_func, test_input[:sample_size, :])
shap_values = explainer.shap_values(test_input[0:sample_size, :], nsamples=200).squeeze()
#shap.force_plot(explainer.expected_value, shap_values, X_display.iloc[299, :])

print()
print()

# print() 
# print("Shap values for ", Output_var)
# print(shap_values)
# print(test_input[0:10, :])
# print(test_output[0:10])


print(shap_values.shape)
# print(test_input[0:10, :].shape)


print() 
print("Linear coefficients for ", Output_var)

coefficients = np.divide(shap_values,(test_input[:sample_size, :]-np.mean(test_input[:sample_size, :], axis = 0)))

avg_coeff = np.mean(coefficients, axis = 0)
variance = np.var(coefficients, axis = 0)
std = np.std(coefficients, axis = 0)

for c, v, s in zip(avg_coeff, variance, std): 
    print(f"{round(c, 5)}, variance {round(v, 5)}, std {round(s, 5)}")




#print(np.divide(shap_values,(test_input[:10, :]-np.mean(test_input[:10, :], axis = 0))))



#print(np.multiply(shap_values,test_input[0, :]))

#shap = coeff_i * (x_i - E[x_i])

#coeff_i = shap/(x_i - E[x_i])