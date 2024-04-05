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

torch.manual_seed(2)
np.random.seed(2)


#regions Parameters
#Hyperparameters
learning_rate = 1e-3
epochs = 1000
Input_scaling = False
Output_scaling = False
Intervene = True
Intervene_info = False
C_D = True

#Output_var = 'y1'
Output_var = 'y2'

print("Complex data")
print("Output variable: ", Output_var)
print("Normalisation: ", Input_scaling, Output_scaling)
print("Intervention: ", Intervene)
print("Intervene Info: ", Intervene_info)
print("Intervene C, D :", C_D)


#region Dataset functions


class MyDataset(Dataset):
    def __init__(self, inputs, targets, output_var):
        self.inputs = inputs
        self.output_var = output_var

        if output_var == 'y1':
            self.targets = targets[:,0]

        elif output_var == 'y2':
            self.targets = targets[:,1]
        else: 
            print("Invalid input variable")
            exit()



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
        scaler_outputs.fit(outputs.reshape(-1, 1))
        self.scaler_outputs = scaler_outputs
        self.targets = torch.from_numpy(self.scaler_outputs.transform(self.targets.reshape(-1, 1)))  #scale all the input features in the dataset, both training and test, according to the training data
        self.targets = self.targets.to(torch.float32).flatten()

        return self.scaler_outputs

    def scale_outputs(self, trained_scaler_outputs):
        self.scaler_outputs = trained_scaler_outputs
        self.targets = torch.from_numpy(self.scaler_outputs.transform(self.targets.reshape(-1, 1)))
        self.targets = self.targets.to(torch.float32).flatten()
    
    
    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.inputs)
    

#endregion


#region ML model functions

# Basic Neural Network
class TestModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TestModel, self).__init__() 


        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        # Apply Kaiming He initialization to all Linear layers in the Sequential model
        self.linear_relu_stack.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)


    
    def forward(self, x):
        x = torch.Tensor(x)
        x = self.linear_relu_stack(x)
        x = x.flatten()

        return x
    
#endregion

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

    #inputs, outputs = Franke_data(n_datapoints, 0)

    #inputs, outputs = super_simple(n_datapoints)


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


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = True)


    ### Define a dataset with different seed
    n_diff_seed = 500
    if Intervene:
        diff_seed_inputs, diff_seed_targets = scm_intv_diff_seed(n_diff_seed, Intervene_info)
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



    #endregion


    #region Create ML Model


    #Network parameters
    input_size = inputs.shape[1]
    output_size = 1 
    hidden_size = 120


    #Model, Loss and Optimizer
    model = TestModel(input_size, hidden_size, output_size)
    loss_fn = nn.MSELoss() 
    optimizer = Adam(model.parameters(), lr=learning_rate)

    #endregion

    #region Train model

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


    file = open(f"progress/{filename}.csv", "w")
    file.write("Epoch,train_loss,val_loss,test_loss,diff_seed_loss,out_of_domain_loss,diff_model_loss,diff_mod_rand_loss,intv_test_loss \n")
    file.close()


    with tqdm(range(epochs + 1), desc='Epochs', unit='epoch', leave=True, bar_format='{desc}: {percentage:3.0f}%|{bar}|{postfix}') as t:
        best_val_loss = 500000
        best_epoch = 0

        for epoch in t:

            model.train()
            running_loss = 0 
            for inputs, targets in train_loader:
                #Forward pass
                optimizer.zero_grad()
                pred = model(inputs)
                loss = loss_fn(pred, targets) 

                # Backward pass
                loss.backward()
                optimizer.step()
                running_loss += loss.item()*100

                

            t.set_postfix(Loss=f'{(running_loss/len(train_loader)):.6f}') 
            t.refresh() # Refresh to show postfix

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_pred = model(val_inputs) 
                    loss = loss_fn(val_pred, val_targets) 
                    val_loss += loss.item()*100

            current_val_loss = val_loss/len(val_loader)

            test_loss = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    pred = model(inputs)
                    loss = loss_fn(pred, targets) 
                    test_loss += loss.item()*100

            diff_seed_loss = 0
            with torch.no_grad():
                for inputs, targets in diff_seed_test_loader:
                    pred = model(inputs)
                    loss = loss_fn(pred, targets) 
                    diff_seed_loss += loss.item()*100


            ood_loss = 0
            with torch.no_grad():
                for inputs, targets in ood_test_loader:
                    pred = model(inputs)
                    loss = loss_fn(pred, targets) 
                    ood_loss += loss.item()*100

            diff_mod_loss = 0
            with torch.no_grad():
                for inputs, targets in diff_mod_loader:
                    pred = model(inputs)
                    loss = loss_fn(pred, targets) 
                    diff_mod_loss += loss.item()*100

            diff_mod_rand_loss = 0
            with torch.no_grad():
                for inputs, targets in diff_mod_rand_loader:
                    pred = model(inputs)
                    loss = loss_fn(pred, targets) 
                    diff_mod_rand_loss += loss.item()*100


            intv_test_loss = 0
            with torch.no_grad():
                for inputs, targets in intv_test_loader:
                    pred = model(inputs)
                    loss = loss_fn(pred, targets) 
                    intv_test_loss += loss.item()*100


            file = open(f"progress/{filename}.csv", "a")
            file.write(f"{epoch},{running_loss/len(train_loader)},{val_loss/len(val_loader)},{test_loss/len(test_loader)},{diff_seed_loss/len(diff_seed_test_loader)},{ood_loss/len(ood_test_loader)},{diff_mod_loss/len(diff_mod_loader)},{diff_mod_rand_loss/len(diff_mod_rand_loader)},{intv_test_loss/len(intv_test_loader)} \n")
            file.close()

            if current_val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = current_val_loss
                train_loss_best_model = running_loss/len(train_loader) 
                save_filename = f"saved_models/best_{filename}.pth"
                torch.save(model, save_filename)

            if epoch % 100 == 0:

                print(f'Epoch {epoch+1}/{epochs} \nTraining Loss: {running_loss/len(train_loader):.6f} '
                f'\nValidation Loss: {val_loss/len(val_loader):.6f}')
                print()

    print()
    print("Finished training")
    print(f'Final val Loss: {val_loss/len(val_loader):.6f}')

    #endregion


    #region Test with other data
    model = torch.load(f"saved_models/best_{filename}.pth")

    print("Best epoch: ", best_epoch)
    print(f'Best val loss: {best_val_loss:.6f}')

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            pred = model(inputs)
            loss = loss_fn(pred, targets) 
            test_loss += loss.item()*100

    print(f'Test loss: {test_loss/len(test_loader):.6f}')

    #print("Test with different seed")
    model.eval()
    diff_seed_loss = 0

    with torch.no_grad():
        for inputs, targets in diff_seed_test_loader:
            pred = model(inputs)
            loss = loss_fn(pred, targets) 
            diff_seed_loss += loss.item()*100

    print(f'Different seed loss: {diff_seed_loss/len(diff_seed_test_loader):.6f}')


    #print("Test with out-of-domain data")

    model.eval()
    ood_loss = 0

    with torch.no_grad():
        for inputs, targets in ood_test_loader:
            pred = model(inputs)
            loss = loss_fn(pred, targets) 
            ood_loss += loss.item()*100

    print(f'Out of domain loss: {ood_loss/len(ood_test_loader):.6f}')
    print()

    diff_mod_loss = 0
    with torch.no_grad():
        for inputs, targets in diff_mod_loader:
            pred = model(inputs)
            loss = loss_fn(pred, targets) 
            diff_mod_loss += loss.item()*100

    print(f'Different model loss: {diff_mod_loss/len(diff_mod_loader):.6f}')
    print()

    diff_mod_rand_loss = 0
    with torch.no_grad():
        for inputs, targets in diff_mod_rand_loader:
            pred = model(inputs)
            loss = loss_fn(pred, targets) 
            diff_mod_rand_loss += loss.item()*100

    print(f'Different model random loss: {diff_mod_rand_loss/len(diff_mod_rand_loader):.6f}')
    print()


    intv_test_loss = 0
    with torch.no_grad():
        for inputs, targets in intv_test_loader:
            pred = model(inputs)
            loss = loss_fn(pred, targets) 
            intv_test_loss += loss.item()*100

    print(f'Interventional test loss: {intv_test_loss/len(intv_test_loader):.6f}')
    print()



    file = open(f"progress/{filename}.csv", "a")
    file.write(f"{best_epoch},{train_loss_best_model},{best_val_loss},{test_loss/len(test_loader)},{diff_seed_loss/len(diff_seed_test_loader)},{ood_loss/len(ood_test_loader)},{diff_mod_loss/len(diff_mod_loader)},{diff_mod_rand_loss/len(diff_mod_rand_loader)},{intv_test_loss/len(intv_test_loader)} \n")
    file.close()

    #endregion


