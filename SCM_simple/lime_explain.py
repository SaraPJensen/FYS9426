import torch 
import torch.nn as nn
import numpy as np
from scm_simple_dataset import scm_dataset_gen, scm_out_of_domain, scm_diff_seed, Franke_data, super_simple, scm_diff_model, scm_diff_rand_model
from scm_intv_simple_dataset import scm_intv_dataset_gen, scm_intv_diff_seed, scm_intv_ood, scm_intv_c_d_dataset_gen
from torch.utils.data import Dataset
import torch.nn.init as init
import lime
from lime import lime_tabular
from sklearn.preprocessing import MinMaxScaler
from scm_simple_network import TestModel


torch.manual_seed(2)
np.random.seed(2)


learning_rate = 1e-3
Input_scaling = False
Output_scaling = False
Intervene = True
Intervene_info = False
C_D = False

#region Dataset functions


class MyDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def fit_scaling_inputs(self, scaler_inputs, training_data):
        inputs, _ = training_data[:]
        scaler_inputs.fit(inputs)
        self.scaler_inputs = scaler_inputs
        self.inputs = torch.from_numpy(self.scaler_inputs.transform(self.inputs))  #scale all the input features in the dataset, both training and test, according to the training data
        self.inputs = self.inputs.to(torch.float32)

        return self.scaler_inputs

    def scale_inputs(self, trained_scaler_inputs):
        self.scaler_inputs = trained_scaler_inputs
        self.inputs = torch.from_numpy(self.scaler_inputs.transform(self.inputs))
        self.inputs = self.inputs.to(torch.float32)



    def fit_scaling_outputs(self, scaler_outputs, training_data):
        _, outputs = training_data[:]
        scaler_outputs.fit(outputs)
        self.scaler_outputs = scaler_outputs
        self.targets = torch.from_numpy(self.scaler_outputs.transform(self.targets))  #scale all the input features in the dataset, both training and test, according to the training data
        self.targets = self.targets.to(torch.float32)

        return self.scaler_outputs

    def scale_outputs(self, trained_scaler_outputs):
        self.scaler_outputs = trained_scaler_outputs
        self.targets = torch.from_numpy(self.scaler_outputs.transform(self.targets))
        self.targets = self.targets.to(torch.float32)
    
        
    
    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.inputs)
    
# endregion

#region Dataset

n_datapoints = 100
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

if Intervene:
    if C_D:
        inputs, outputs = scm_intv_c_d_dataset_gen(n_datapoints, Intervene_info)
    else:
        inputs, outputs = scm_intv_dataset_gen(n_datapoints, Intervene_info)

else:
    inputs, outputs = scm_dataset_gen(n_datapoints)

#inputs, outputs = Franke_data(n_datapoints, 0)

#inputs, outputs = super_simple(n_datapoints)


batch_size = 32

#Make torch dataset
input_tensor = torch.from_numpy(inputs).float()
output_tensor = torch.from_numpy(outputs).float()


torch_dataset = MyDataset(input_tensor, output_tensor) 
# This implicitly shuffles the data as well, do don't need to include that in the dataset generator for the interventional data
train_data, val_data, test_data = torch.utils.data.random_split(torch_dataset, [int(0.7*n_datapoints), int(0.15*n_datapoints), int(0.15*n_datapoints)])  

if Input_scaling:
    trained_scaler_inputs = torch_dataset.fit_scaling_inputs(input_scaler, train_data)

if Output_scaling:
    trained_scaler_outputs = torch_dataset.fit_scaling_outputs(output_scaler, train_data)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = True)


### Define a dataset with different seed
n_diff_seed = 50
if Intervene:
    diff_seed_inputs, diff_seed_targets = scm_intv_diff_seed(n_diff_seed, Intervene_info)
else:
    diff_seed_inputs, diff_seed_targets = scm_diff_seed(n_diff_seed)

diff_input_tensor = torch.from_numpy(diff_seed_inputs).float()
diff_output_tensor = torch.from_numpy(diff_seed_targets).float()

diff_seed_torch_dataset = MyDataset(diff_input_tensor, diff_output_tensor) 
if Input_scaling:
    diff_seed_torch_dataset.scale_inputs(trained_scaler_inputs)
if Output_scaling:
    diff_seed_torch_dataset.scale_outputs(trained_scaler_outputs)
diff_seed_test_loader = torch.utils.data.DataLoader(diff_seed_torch_dataset, batch_size = 1, shuffle = True)


### Define an out-of-domain dataset
n_ood = 50

if Intervene:
    ood_inputs, ood_targets = scm_intv_ood(n_ood, Intervene_info)
else:
    ood_inputs, ood_targets = scm_out_of_domain(n_ood)


input_tensor = torch.from_numpy(ood_inputs).float()
output_tensor = torch.from_numpy(ood_targets).float()

ood_torch_dataset = MyDataset(input_tensor, output_tensor) 
if Input_scaling:
    ood_torch_dataset.scale_inputs(trained_scaler_inputs)
if Output_scaling:
    ood_torch_dataset.scale_outputs(trained_scaler_outputs)
ood_test_loader = torch.utils.data.DataLoader(ood_torch_dataset, batch_size = 1, shuffle = True)


### Define a dataset with different SCM between the inputs
#For now, this is never interventional 

n_diff_model = 50
diff_mod_inputs, diff_mod_targets = scm_diff_model(n_diff_model, Intervene_info)

input_tensor = torch.from_numpy(diff_mod_inputs).float()
output_tensor = torch.from_numpy(diff_mod_targets).float()

diff_mod_torch_dataset = MyDataset(input_tensor, output_tensor) 
if Input_scaling:
    diff_mod_torch_dataset.scale_inputs(trained_scaler_inputs)
if Output_scaling:
    diff_mod_torch_dataset.scale_outputs(trained_scaler_outputs)
diff_mod_loader = torch.utils.data.DataLoader(diff_mod_torch_dataset, batch_size = 1, shuffle = True)


### Define a dataset with different SCM between the inputs
#For now, this is never interventional 

n_diff_rand_model = 50
diff_rand_mod_inputs, diff_rand_mod_targets = scm_diff_rand_model(n_diff_rand_model, Intervene_info)

input_tensor = torch.from_numpy(diff_rand_mod_inputs).float()
output_tensor = torch.from_numpy(diff_rand_mod_targets).float()

diff_mod_rand_torch_dataset = MyDataset(input_tensor, output_tensor) 
if Input_scaling:
    diff_mod_rand_torch_dataset.scale_inputs(trained_scaler_inputs)
if Output_scaling:
    diff_mod_rand_torch_dataset.scale_outputs(trained_scaler_outputs)
diff_mod_rand_loader = torch.utils.data.DataLoader(diff_mod_rand_torch_dataset, batch_size = 1, shuffle = True)


#Define a dataset with interventional data, using a different seed from the training
n_intv_testing = 50
intv_inputs, intv_targets = scm_intv_dataset_gen(n_intv_testing, Intervene_info, seed = 54321)

input_tensor = torch.from_numpy(intv_inputs).float()
output_tensor = torch.from_numpy(intv_targets).float()

intv_torch_dataset = MyDataset(input_tensor, output_tensor) 
if Input_scaling:
    intv_torch_dataset.scale_inputs(trained_scaler_inputs)
if Output_scaling:
    intv_torch_dataset.scale_outputs(trained_scaler_outputs)
intv_test_loader = torch.utils.data.DataLoader(intv_torch_dataset, batch_size = 1, shuffle = True)



def extract_tensors(data_loader):
    X = []
    y = []
    for batch in data_loader:
        inputs, targets = batch
        X.append(inputs.numpy())
        y.append(targets.numpy())
    return np.concatenate(X), np.concatenate(y)

inputs_test_np, outputs_test_np = extract_tensors(test_loader)

#endregion




#region Load model
if Input_scaling and Output_scaling:
    filename = f"MinMax_all_lr_{learning_rate}_train_test_loss"
elif Input_scaling:
    filename = f"MinMax_inputs_raw_outputs_lr_{learning_rate}_train_test_loss"
elif Output_scaling:
    filename = f"MinMax_outputs_raw_inputs_lr_{learning_rate}_train_test_loss"
else:
    filename = f"Raw_data_lr_{learning_rate}_train_test_loss"

if Intervene:
    if C_D:
        if Intervene_info: 
            filename = "Intv_C_D_Info_" + filename
        else:
            filename = "Intv_C_D_noInfo_" + filename

    else:
        if Intervene_info:
            filename = "Intv_Info_" + filename
        else:
            filename = "Intv_noInfo_" + filename


model = torch.load(f"saved_models/best_{filename}.pth")
model.eval()

#endregion


def true_func_1(inputs):
    y1 = 3.5*inputs[:,4]
    #y2 = 4*inputs[:,0]

    #out = np.column_stack((y1, y2))

    return y1

def true_func_2(inputs):
    #y1 = 3.5*inputs[:,4]
    y2 = 4*inputs[:,3]

    #out = np.column_stack((y1, y2))

    return y2


#region LIME
input_feature_names = ['A', 'B', 'C', 'D', 'E']
output_feature_names = ['y1', 'y2']

with torch.no_grad():
    explainer_1 = lime_tabular.LimeTabularExplainer(inputs_test_np, 
                                                  mode='regression', 
                                                  feature_names = input_feature_names)
    
    explainer_2 = lime_tabular.LimeTabularExplainer(inputs_test_np, 
                                                  mode='regression', 
                                                  feature_names = input_feature_names)

    #selected_instances = torch.Tensor(inputs_test_np[0])
    selected_instances = inputs_test_np[0]

    # explanation_1 = explainer.explain_instance(selected_instances, 
    #                                          true_func_1, 
    #                                          num_features = len(input_feature_names))
    
    # explanation_2 = explainer.explain_instance(selected_instances, 
    #                                          true_func_2, 
    #                                          num_features = len(input_feature_names))

    explanation_y1 = explainer_1.explain_instance(selected_instances, 
                                                   true_func_1, 
                                                   num_features=len(input_feature_names))

    explanation_y2 = explainer_2.explain_instance(selected_instances,
                                                    true_func_2, 
                                                    num_features=len(input_feature_names))


print(inputs_test_np[0])
print()
print(outputs_test_np[0])

print()

print('y1')
print(explanation_y1.as_list())

print()
print('y2')
print(explanation_y2.as_list())

 

