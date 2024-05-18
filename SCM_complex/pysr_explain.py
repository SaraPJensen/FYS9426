from pysr import PySRRegressor
import torch 
import torch.nn as nn
import numpy as np
from scm_complex_dataset import scm_dataset_gen, scm_out_of_domain, scm_diff_seed, scm_diff_model, scm_diff_rand_model, scm_indep_ood, scm_normal_dist, scm_indep
from scm_intv_complex_dataset import scm_intv_dataset_gen, scm_intv_ood, scm_intv_c_d_dataset_gen
from filename_funcs import get_filename, get_model_name
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
from scm_complex_network import MyDataset, DeepModel, TestModel
import os
import dill

import sklearn
import shap


torch.manual_seed(2)
np.random.seed(2)

True_model = False
Scaling = False
Deep = True

Intervene = True 
C_D = False
Independent = False
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


batch_size = 32

#Make torch dataset - this is just for scaling
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
    obsv_torch_dataset.scale_inputs(trained_scaler_inputs)
    obsv_torch_dataset.scale_outputs(trained_scaler_outputs)
obsv_torch_loader = torch.utils.data.DataLoader(obsv_torch_dataset, batch_size=1, shuffle=True)


#Define a dataset with interventional data, using a different seed from the training
n_intv_testing = 500
intv_inputs, intv_targets = scm_intv_dataset_gen(n_intv_testing, seed = 54321)
input_tensor = torch.from_numpy(intv_inputs).float()
output_tensor = torch.from_numpy(intv_targets).float()
intv_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    intv_torch_dataset.scale_inputs(trained_scaler_inputs)
    intv_torch_dataset.scale_outputs(trained_scaler_outputs)
intv_torch_loader = torch.utils.data.DataLoader(intv_torch_dataset, batch_size=1, shuffle=True)



### Define an out-of-domain dataset with observational data
n_ood = 500

ood_inputs, ood_targets = scm_out_of_domain(n_ood)
input_tensor = torch.from_numpy(ood_inputs).float()
output_tensor = torch.from_numpy(ood_targets).float()
ood_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    ood_torch_dataset.scale_inputs(trained_scaler_inputs)
    ood_torch_dataset.scale_outputs(trained_scaler_outputs)
ood_torch_loader = torch.utils.data.DataLoader(ood_torch_dataset, batch_size=1, shuffle=True)

### Define an out-of-domain dataset with interventional data
ood_intv_inputs, ood_intv_targets = scm_intv_ood(n_ood)
input_tensor = torch.from_numpy(ood_intv_inputs).float()
output_tensor = torch.from_numpy(ood_intv_targets).float()
ood_intv_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    ood_intv_torch_dataset.scale_inputs(trained_scaler_inputs)
    ood_intv_torch_dataset.scale_outputs(trained_scaler_outputs)
ood_intv_torch_loader = torch.utils.data.DataLoader(ood_intv_torch_dataset, batch_size=1, shuffle=True)


### Define a dataset with different SCM between the inputs
n_diff_model = 500
diff_mod_inputs, diff_mod_targets = scm_diff_model(n_diff_model)
input_tensor = torch.from_numpy(diff_mod_inputs).float()
output_tensor = torch.from_numpy(diff_mod_targets).float()
diff_mod_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    diff_mod_torch_dataset.scale_inputs(trained_scaler_inputs)
    diff_mod_torch_dataset.scale_outputs(trained_scaler_outputs)
diff_mod_torch_loader = torch.utils.data.DataLoader(diff_mod_torch_dataset, batch_size=1, shuffle=True)

### Define a dataset with different SCM between the inputs, here the inputs are independent of each other
n_diff_rand_model = 500
diff_rand_mod_inputs, diff_rand_mod_targets = scm_diff_rand_model(n_diff_rand_model)
input_tensor = torch.from_numpy(diff_rand_mod_inputs).float()
output_tensor = torch.from_numpy(diff_rand_mod_targets).float()
diff_mod_rand_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
if Scaling:
    diff_mod_rand_torch_dataset.scale_inputs(trained_scaler_inputs)
    diff_mod_rand_torch_dataset.scale_outputs(trained_scaler_outputs)
diff_mod_rand_torch_loader = torch.utils.data.DataLoader(diff_mod_rand_torch_dataset, batch_size=1, shuffle=True)



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


filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
model_name = get_model_name(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
trained_model = torch.load(model_name)
trained_model.eval()

if True_model:
    filename = "true_model"
    trained_model = true_model

save_path = f"pysr/{Output_var}/{filename}/"
save_file = save_path + "best_funcs.csv"

if not os.path.exists(save_path):
    os.makedirs(save_path)


file = open(save_file, "w")
file.write("Dataset,expression\n")
file.close()


feature_names = ['a', 'b', 'c', 'd', 'e']


# Generate dataset for the model predictions of each model. Must rescale both the inputs and the outputs before passing them to PySR

token_inputs, token_outputs = obsv_torch_dataset[:]

def fit_pysr(dataset_name, trained_model, dataloader, save_path, save_file, Scaling, token_inputs, token_outputs):

    pysr_inputs = torch.zeros_like(token_inputs)
    pysr_preds = torch.zeros_like(token_outputs)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            pred = trained_model(inputs)
            if Scaling:
                inputs = obsv_torch_dataset.rescale_inputs(inputs)
                preds = torch_dataset.rescale_outputs(pred)

            
            pysr_inputs[i,:] = inputs
            pysr_preds[i,:] = pred

    model = PySRRegressor(
        niterations=10,  # < Increase me for better results
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "square",
            "inv(x) = 1/x",
        ],
        constraints = {'^': (-1, 1)},
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        equation_file = f"{save_path}_{dataset_name}.csv",
    )

    model.fit(pysr_inputs.numpy(), pysr_preds.numpy(), variable_names = feature_names)

    best_eq = model.get_best().equation

    #Now pass the dataset through the best equation to test the fit compared with the original dataset
    file = open(save_file, "w")
    file.write(f"{dataset_name},{best_eq}\n")
    file.close()


fit_pysr("obsv", trained_model, obsv_torch_loader, save_path, save_file, Scaling, token_inputs, token_outputs)

fit_pysr("intv", trained_model, intv_torch_loader, save_path, save_file, Scaling, token_inputs, token_outputs)

fit_pysr("ood", trained_model, ood_torch_loader, save_path, save_file, Scaling, token_inputs, token_outputs)

fit_pysr("ood_intv", trained_model, ood_intv_torch_loader, save_path, save_file, Scaling, token_inputs, token_outputs)

fit_pysr("diff_mod", trained_model, diff_mod_torch_loader, save_path, save_file, Scaling, token_inputs, token_outputs)

fit_pysr("diff_mod_rand", trained_model, diff_mod_rand_torch_loader, save_path, save_file, Scaling, token_inputs, token_outputs)

exit()



test_input, test_output = test_data[:]
test_input = np.asarray(test_input.numpy())
test_output = np.asarray(test_output.numpy())


intv_input, intv_output = intv_torch_dataset[:]
intv_input = np.asarray(intv_input.numpy())
intv_output = np.asarray(intv_output.numpy())


model = PySRRegressor(
    niterations=10,  # < Increase me for better results
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "square",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    constraints = {'^': (-1, 1)},
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    equation_file = f"{save_path}eqs.csv",
    # tempdir = "pysr/y1/temp",
    # temp_equation_file = True,
    # delete_tempfiles = False
)


model.fit(intv_input, intv_output, variable_names = feature_names)

print("Final expression: ", model)

print("Best equation: ", model.get_best().equation) 

print()

d_point = np.asarray([1, 2, 3, 4, 5]).reshape(1, -1)

print(model.predict(d_point))


symbol = str(model.sympy())
print("String of sympy: ", symbol)

save_symbol_filename = f"{save_path}best_eq.csv"
save_symbol_file = open(save_symbol_filename, "w")
save_symbol_file.write(symbol)
save_symbol_file.close()


# symbol = model.pytorch()
# print(symbol)
# print(symbol(d_point))





