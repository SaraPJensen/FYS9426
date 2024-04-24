import torch 
import torch.nn as nn
import numpy as np
from scm_complex_dataset import scm_dataset_gen, scm_out_of_domain, scm_diff_seed, Franke_data, super_simple, scm_diff_model, scm_diff_rand_model
from scm_intv_complex_dataset import scm_intv_dataset_gen, scm_intv_diff_seed, scm_intv_ood, scm_intv_c_d_dataset_gen
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.init as init
from lime import lime_tabular
from sklearn.preprocessing import MinMaxScaler
from scm_complex_network import MyDataset, TestModel, DeepModel



torch.manual_seed(2)
np.random.seed(2)


#regions Parameters
#Hyperparameters
learning_rate = 1e-3
epochs = 1000
Input_scaling = True
Output_scaling = True
Intervene = False
Intervene_info = False
C_D = False
Deep = False
Rescale = True

#Output_var = 'y1'
Output_var = 'y2'


print("Complex data")
print("Model trained on:")
print("Output variable: ", Output_var)
print("Normalisation: ", Input_scaling, Output_scaling)
print("Intervention: ", Intervene)
print("Intervene Info: ", Intervene_info)
print("Intervene C, D :", C_D)
print("Deep: ", Deep)


if __name__ == "__main__":

    #region Make dataset 

    n_datapoints = 3000
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

    if Input_scaling:
        trained_scaler_inputs = torch_dataset.fit_scaling_inputs(input_scaler, train_data)

    if Output_scaling:
        trained_scaler_outputs = torch_dataset.fit_scaling_outputs(output_scaler, train_data)

    #This is equavalent to the original dataset
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = True)


    ### Define a dataset with different seed
    n_diff_seed = 500
    if Intervene:
        if C_D:
            diff_seed_inputs, diff_seed_targets = scm_intv_c_d_dataset_gen(n_datapoints, Intervene_info, seed = 12345)
        else:
            diff_seed_inputs, diff_seed_targets = scm_intv_dataset_gen(n_datapoints, Intervene_info, seed = 12345)
    else:
        diff_seed_inputs, diff_seed_targets = scm_diff_seed(n_diff_seed)
    
    diff_input_tensor = torch.from_numpy(diff_seed_inputs).float()
    diff_output_tensor = torch.from_numpy(diff_seed_targets).float()

    diff_seed_torch_dataset = MyDataset(diff_input_tensor, diff_output_tensor, Output_var) 
    if Input_scaling:
        diff_seed_torch_dataset.scale_inputs(trained_scaler_inputs)
    if Output_scaling:
        diff_seed_torch_dataset.scale_outputs(trained_scaler_outputs)
    diff_seed_test_loader = torch.utils.data.DataLoader(diff_seed_torch_dataset, batch_size = 1, shuffle = True)


    ### Define an out-of-domain dataset
    n_ood = 500

    if Intervene:
        ood_inputs, ood_targets = scm_intv_ood(n_ood, Intervene_info)
    else:
        ood_inputs, ood_targets = scm_out_of_domain(n_ood)


    input_tensor = torch.from_numpy(ood_inputs).float()
    output_tensor = torch.from_numpy(ood_targets).float()

    ood_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var) 
    if Input_scaling:
        ood_torch_dataset.scale_inputs(trained_scaler_inputs)
    if Output_scaling:
        ood_torch_dataset.scale_outputs(trained_scaler_outputs)
    ood_test_loader = torch.utils.data.DataLoader(ood_torch_dataset, batch_size = 1, shuffle = True)


    ### Define a dataset with different SCM between the inputs
    #For now, this is never interventional 

    n_diff_model = 500
    diff_mod_inputs, diff_mod_targets = scm_diff_model(n_diff_model, Intervene_info)

    input_tensor = torch.from_numpy(diff_mod_inputs).float()
    output_tensor = torch.from_numpy(diff_mod_targets).float()

    diff_mod_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var) 
    if Input_scaling:
        diff_mod_torch_dataset.scale_inputs(trained_scaler_inputs)
    if Output_scaling:
        diff_mod_torch_dataset.scale_outputs(trained_scaler_outputs)
    diff_mod_loader = torch.utils.data.DataLoader(diff_mod_torch_dataset, batch_size = 1, shuffle = True)


    ### Define a dataset with different SCM between the inputs
    #For now, this is never interventional 

    n_diff_rand_model = 500
    diff_rand_mod_inputs, diff_rand_mod_targets = scm_diff_rand_model(n_diff_rand_model, Intervene_info)

    input_tensor = torch.from_numpy(diff_rand_mod_inputs).float()
    output_tensor = torch.from_numpy(diff_rand_mod_targets).float()

    diff_mod_rand_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var) 
    if Input_scaling:
        diff_mod_rand_torch_dataset.scale_inputs(trained_scaler_inputs)
    if Output_scaling:
        diff_mod_rand_torch_dataset.scale_outputs(trained_scaler_outputs)
    diff_mod_rand_loader = torch.utils.data.DataLoader(diff_mod_rand_torch_dataset, batch_size = 1, shuffle = True)


    #Define a dataset with interventional data, using a different seed from the training
    n_intv_testing = 500
    intv_inputs, intv_targets = scm_intv_dataset_gen(n_intv_testing, Intervene_info, seed = 54321)

    input_tensor = torch.from_numpy(intv_inputs).float()
    output_tensor = torch.from_numpy(intv_targets).float()

    intv_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var) 
    if Input_scaling:
        intv_torch_dataset.scale_inputs(trained_scaler_inputs)
    if Output_scaling:
        intv_torch_dataset.scale_outputs(trained_scaler_outputs)
    intv_test_loader = torch.utils.data.DataLoader(intv_torch_dataset, batch_size = 1, shuffle = True)


    #Define a dataset with observational data, using a different seed from the training
    n_obsv_testing = 500
    obsv_inputs, obsv_targets = scm_dataset_gen(n_obsv_testing, seed = 54321)

    input_tensor = torch.from_numpy(obsv_inputs).float()
    output_tensor = torch.from_numpy(obsv_targets).float()

    obsv_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var) 
    if Input_scaling:
        obsv_torch_dataset.scale_inputs(trained_scaler_inputs)
    if Output_scaling:
        obsv_torch_dataset.scale_outputs(trained_scaler_outputs)
    obsv_test_loader = torch.utils.data.DataLoader(obsv_torch_dataset, batch_size = 1, shuffle = True)


    #endregion


    #region Create ML Model

    #Network parameters
    input_size = inputs.shape[1]
    output_size = 1 
    hidden_size = 120


    #Model, Loss and Optimizer
    if Deep:
        model = DeepModel(input_size, hidden_size, output_size)
    else:
        model = TestModel(input_size, hidden_size, output_size)
    loss_fn = nn.MSELoss() 

    #endregion


    file_intv = ''
    if Intervene:
        if C_D:
            if Intervene_info: 
                file_intv = "Intv_C_D_Info_"
            else:
                file_intv = "Intv_C_D_noInfo_"

        else:
            if Intervene_info:
                file_intv = "Intv_Info_"
            else:
                file_intv = "Intv_noInfo_"


    if Input_scaling and Output_scaling:
        filename = f"{Output_var}_{file_intv}MinMax_all_lr_{learning_rate}"
    elif Input_scaling:
        filename = f"{Output_var}_{file_intv}MinMax_inputs_raw_outputs_lr_{learning_rate}"
    elif Output_scaling:
        filename = f"{Output_var}_{file_intv}MinMax_outputs_raw_inputs_lr_{learning_rate}"
    else:
        filename = f"{Output_var}_{file_intv}Raw_data_lr_{learning_rate}"

    if Deep:
        filename = "deeeeep_" + filename

    if Rescale: 
        add = "rescale_" 
    else: 
        add = ''

    file = open(f"progress/{Output_var}/test_model/{add}{filename}.csv", "w")
    file.write("train_loss,val_loss,test_loss,diff_seed_loss,out_of_domain_loss,diff_model_loss,diff_mod_rand_loss,obsv_test_loss,intv_test_loss \n")
    file.close()


    
    #region Test with other data
    model = torch.load(f"saved_models/{Output_var}/best_{filename}.pth")


    model.eval()
    train_loss = 0

    with torch.no_grad():
        for inputs, targets in train_loader:
            pred = model(inputs)
            if Rescale: 
                # print(pred.shape)
                # print(targets.shape)
                pred = torch_dataset.rescale_outputs(pred)
                targets = torch_dataset.rescale_outputs(targets)
                # print(pred.shape)
                # print(targets.shape)
                # exit()
                loss = loss_fn(pred, targets)
            else:
                loss = loss_fn(pred, targets) 
            train_loss += loss.item()*100


    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            pred = model(inputs)
            if Rescale: 
                pred = torch_dataset.rescale_outputs(pred)
                targets = torch_dataset.rescale_outputs(targets)
                loss = loss_fn(pred, targets)
            else:
                loss = loss_fn(pred, targets) 
            val_loss += loss.item()*100


    model.eval()
    test_loss = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            pred = model(inputs)
            if Rescale: 
                pred = torch_dataset.rescale_outputs(pred)
                targets = torch_dataset.rescale_outputs(targets)
                loss = loss_fn(pred, targets)
            else:
                loss = loss_fn(pred, targets) 
            test_loss += loss.item()*100


    model.eval()
    diff_seed_loss = 0

    with torch.no_grad():
        for inputs, targets in diff_seed_test_loader:
            pred = model(inputs)
            if Rescale: 
                pred = diff_seed_torch_dataset.rescale_outputs(pred)
                targets = diff_seed_torch_dataset.rescale_outputs(targets)
                loss = loss_fn(pred, targets)
            else:
                loss = loss_fn(pred, targets) 
            diff_seed_loss += loss.item()*100


    model.eval()
    ood_loss = 0

    with torch.no_grad():
        for inputs, targets in ood_test_loader:
            pred = model(inputs)
            if Rescale: 
                pred = ood_torch_dataset.rescale_outputs(pred)
                targets = ood_torch_dataset.rescale_outputs(targets)
                loss = loss_fn(pred, targets)
            else:
                loss = loss_fn(pred, targets) 
            ood_loss += loss.item()*100


    diff_mod_loss = 0
    with torch.no_grad():
        for inputs, targets in diff_mod_loader:
            pred = model(inputs)
            if Rescale: 
                pred = diff_mod_torch_dataset.rescale_outputs(pred)
                targets = diff_mod_torch_dataset.rescale_outputs(targets)
                loss = loss_fn(pred, targets)
            else:
                loss = loss_fn(pred, targets) 
            diff_mod_loss += loss.item()*100


    diff_mod_rand_loss = 0
    with torch.no_grad():
        for inputs, targets in diff_mod_rand_loader:
            pred = model(inputs)
            if Rescale: 
                pred = diff_mod_rand_torch_dataset.rescale_outputs(pred)
                targets = diff_mod_rand_torch_dataset.rescale_outputs(targets)
                loss = loss_fn(pred, targets)
            else:
                loss = loss_fn(pred, targets) 
            diff_mod_rand_loss += loss.item()*100


    obsv_test_loss = 0
    with torch.no_grad():
        for inputs, targets in obsv_test_loader:
            pred = model(inputs)
            if Rescale: 
                pred = obsv_torch_dataset.rescale_outputs(pred)
                targets = obsv_torch_dataset.rescale_outputs(targets)
                loss = loss_fn(pred, targets)
            else:
                loss = loss_fn(pred, targets)  
            obsv_test_loss += loss.item()*100


    intv_test_loss = 0
    with torch.no_grad():
        for inputs, targets in intv_test_loader:
            pred = model(inputs)
            if Rescale: 
                pred = intv_torch_dataset.rescale_outputs(pred)
                targets = intv_torch_dataset.rescale_outputs(targets)
                loss = loss_fn(pred, targets)
            else:
                loss = loss_fn(pred, targets) 
            intv_test_loss += loss.item()*100


    file = open(f"progress/{Output_var}/test_model/{add}{filename}.csv", "a")
    file.write(f"{train_loss/len(train_loader)},{val_loss/len(val_loader)},{test_loss/len(test_loader)},{diff_seed_loss/len(diff_seed_test_loader)},{ood_loss/len(ood_test_loader)},{diff_mod_loss/len(diff_mod_loader)},{diff_mod_rand_loss/len(diff_mod_rand_loader)},{obsv_test_loss/len(obsv_test_loader)},{intv_test_loss/len(intv_test_loader)} \n")
    file.close()


