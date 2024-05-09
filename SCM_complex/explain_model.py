import torch 
import torch.nn as nn
import numpy as np
from scm_complex_dataset import scm_dataset_gen, scm_out_of_domain, scm_diff_seed, scm_diff_model, scm_diff_rand_model, scm_indep_ood, scm_normal_dist, scm_indep
from scm_intv_complex_dataset import scm_intv_dataset_gen, scm_intv_ood, scm_intv_c_d_dataset_gen
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
from scm_complex_network import MyDataset, TestModel, DeepModel
from alibi.explainers import ALE


torch.manual_seed(2)
np.random.seed(2)


#regions Parameters
#Hyperparameters
learning_rate = 1e-3

Deep = False
Scaling = True 

Intervene = False
C_D = False
Independent = True
Simplify = False

Output_var = 'y1'
#Output_var = 'y2'


print("Complex data")
print("Model trained on:")
print("Output variable: ", Output_var)
print("Normalisation: ", Scaling)
print("Intervention: ", Intervene)
print("Intervene C, D :", C_D)
print("Independent inputs: ", Independent)
print("Deep: ", Deep)
print("Simple: ", Simplify)


if __name__ == "__main__":

    #region Make dataset 

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

    #Make torch dataset
    input_tensor = torch.from_numpy(inputs).float()
    output_tensor = torch.from_numpy(outputs).float()


    torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
    # This implicitly shuffles the data as well, do don't need to include that in the dataset generator for the interventional data
    train_data, val_data, test_data = torch.utils.data.random_split(torch_dataset, [int(0.7*n_datapoints), int(0.15*n_datapoints), int(0.15*n_datapoints)])  

    if Scaling:
        trained_scaler_inputs = torch_dataset.fit_scaling_inputs(input_scaler, train_data)
        trained_scaler_outputs = torch_dataset.fit_scaling_outputs(output_scaler, train_data)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = True)



    ### Define a dataset with different seed
    n_diff_seed = 500
    if Intervene:
        if C_D:
            diff_seed_inputs, diff_seed_targets = scm_intv_c_d_dataset_gen(n_datapoints, seed = 12345)
        else:
            diff_seed_inputs, diff_seed_targets = scm_intv_dataset_gen(n_datapoints, seed = 12345)
    elif Independent:
        diff_seed_inputs, diff_seed_targets = scm_indep(n_diff_seed, seed = 12345)
    else:
        diff_seed_inputs, diff_seed_targets = scm_diff_seed(n_diff_seed)
    
    diff_input_tensor = torch.from_numpy(diff_seed_inputs).float()
    diff_output_tensor = torch.from_numpy(diff_seed_targets).float()

    diff_seed_torch_dataset = MyDataset(diff_input_tensor, diff_output_tensor, Output_var, Simplify) 
    if Scaling:
        diff_seed_torch_dataset.scale_inputs(trained_scaler_inputs)
        diff_seed_torch_dataset.scale_outputs(trained_scaler_outputs)
    diff_seed_test_loader = torch.utils.data.DataLoader(diff_seed_torch_dataset, batch_size = 1, shuffle = True)


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
        ood_torch_dataset.scale_inputs(trained_scaler_inputs)
        ood_torch_dataset.scale_outputs(trained_scaler_outputs)
    ood_test_loader = torch.utils.data.DataLoader(ood_torch_dataset, batch_size = 1, shuffle = True)


    ### Define a dataset with different SCM between the inputs
    #For now, this is never interventional 

    n_diff_model = 500
    diff_mod_inputs, diff_mod_targets = scm_diff_model(n_diff_model)

    input_tensor = torch.from_numpy(diff_mod_inputs).float()
    output_tensor = torch.from_numpy(diff_mod_targets).float()

    diff_mod_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
    if Scaling:
        diff_mod_torch_dataset.scale_inputs(trained_scaler_inputs)
        diff_mod_torch_dataset.scale_outputs(trained_scaler_outputs)
    diff_mod_loader = torch.utils.data.DataLoader(diff_mod_torch_dataset, batch_size = 1, shuffle = True)


    ### Define a dataset with different SCM between the inputs, here the inputs are independent of each other
    #For now, this is never interventional 

    n_diff_rand_model = 500
    diff_rand_mod_inputs, diff_rand_mod_targets = scm_diff_rand_model(n_diff_rand_model)

    input_tensor = torch.from_numpy(diff_rand_mod_inputs).float()
    output_tensor = torch.from_numpy(diff_rand_mod_targets).float()

    diff_mod_rand_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
    if Scaling:
        diff_mod_rand_torch_dataset.scale_inputs(trained_scaler_inputs)
        diff_mod_rand_torch_dataset.scale_outputs(trained_scaler_outputs)
    diff_mod_rand_loader = torch.utils.data.DataLoader(diff_mod_rand_torch_dataset, batch_size = 1, shuffle = True)


    #Define a dataset with interventional data, using a different seed from the training
    n_intv_testing = 500
    intv_inputs, intv_targets = scm_intv_dataset_gen(n_intv_testing, seed = 54321)

    input_tensor = torch.from_numpy(intv_inputs).float()
    output_tensor = torch.from_numpy(intv_targets).float()

    intv_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
    if Scaling:
        intv_torch_dataset.scale_inputs(trained_scaler_inputs)
        intv_torch_dataset.scale_outputs(trained_scaler_outputs)
    intv_test_loader = torch.utils.data.DataLoader(intv_torch_dataset, batch_size = 1, shuffle = True)


    #Define a dataset with observational data, using a different seed from the training
    n_obsv_testing = 500
    obsv_inputs, obsv_targets = scm_dataset_gen(n_obsv_testing, seed = 54321)

    input_tensor = torch.from_numpy(obsv_inputs).float()
    output_tensor = torch.from_numpy(obsv_targets).float()

    obsv_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
    if Scaling:
        obsv_torch_dataset.scale_inputs(trained_scaler_inputs)
        obsv_torch_dataset.scale_outputs(trained_scaler_outputs)
    obsv_test_loader = torch.utils.data.DataLoader(obsv_torch_dataset, batch_size = 1, shuffle = True)



    #Define a dataset with observational data with the same input ranges, but with a normal, rather than uniform distribution
    n_normal_testing = 500
    normal_inputs, normal_targets = scm_normal_dist(n_normal_testing, seed = 5)

    input_tensor = torch.from_numpy(normal_inputs).float()
    output_tensor = torch.from_numpy(normal_targets).float()

    normal_torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
    if Scaling:
        normal_torch_dataset.scale_inputs(trained_scaler_inputs)
        normal_torch_dataset.scale_outputs(trained_scaler_outputs)
    normal_test_loader = torch.utils.data.DataLoader(normal_torch_dataset, batch_size = 1, shuffle = True)




    #endregion


    #region Create ML Model

    #Network parameters
    input_size = inputs.shape[1]
    if Simplify:
        input_size = 3
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
            file_intv = "Intv_C_D_"
        else:
            file_intv = "Intv_"


    if Scaling:
        filename = f"{Output_var}_{file_intv}MinMax_lr_{learning_rate}"
    else:
        filename = f"{Output_var}_{file_intv}Raw_data_lr_{learning_rate}"

    if Independent:
        filename = "indep_" + filename

    if Simplify: 
        filename = "simple_" + filename

    if Deep:
        filename = "deeeeep_" + filename


    
    model = torch.load(f"saved_models/{Output_var}/best_{filename}.pth")


    