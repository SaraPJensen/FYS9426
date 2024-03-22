import torch 
import torch.nn as nn
import numpy as np
from scm_simple_dataset import scm_dataset_gen, scm_out_of_domain, scm_diff_seed, Franke_data, super_simple
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.init as init
from lime import lime_tabular
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(2)
np.random.seed(2)



#region Dataset functions

def normalise_data(inputs, targets):

    # Calculate mean and standard deviation for normalization
    input_mean = inputs.mean(dim=0)
    input_std = inputs.std(dim=0)
    target_mean = targets.mean(dim=0)
    target_std = targets.std(dim=0)


    # Normalise the data
    inputs_normalised = (inputs - input_mean) / input_std
    targets_normalised = (targets - target_mean) / target_std

    return inputs_normalised, targets_normalised



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
        x = self.linear_relu_stack(x)

        """ x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) """

        return x
    
#endregion


#region Make dataset 

n_datapoints = 3000
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()


inputs, outputs = scm_dataset_gen(n_datapoints)

#inputs, outputs = Franke_data(n_datapoints, 0)

#inputs, outputs = super_simple(n_datapoints)


batch_size = 32

#Make torch dataset
input_tensor = torch.from_numpy(inputs).float()
output_tensor = torch.from_numpy(outputs).float()


torch_dataset = MyDataset(input_tensor, output_tensor) 
train_data, val_data, test_data = torch.utils.data.random_split(torch_dataset, [int(0.7*n_datapoints), int(0.15*n_datapoints), int(0.15*n_datapoints)])


trained_scaler_inputs = torch_dataset.fit_scaling_inputs(input_scaler, train_data)
trained_scaler_outputs = torch_dataset.fit_scaling_outputs(output_scaler, train_data)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)


### Define a dataset with different seed
n_diff_seed = 500
diff_seed_inputs, diff_seed_targets = scm_diff_seed(n_diff_seed)

diff_input_tensor = torch.from_numpy(diff_seed_inputs).float()
diff_output_tensor = torch.from_numpy(diff_seed_targets).float()

diff_seed_torch_dataset = MyDataset(diff_input_tensor, diff_output_tensor) 
diff_seed_torch_dataset.scale_inputs(trained_scaler_inputs)
diff_seed_torch_dataset.scale_outputs(trained_scaler_outputs)
diff_seed_test_loader = torch.utils.data.DataLoader(diff_seed_torch_dataset, batch_size = 1, shuffle = False)


### Define an out-of-domain dataset
n_ood = 500
ood_inputs, ood_targets = scm_out_of_domain(n_ood)

input_tensor = torch.from_numpy(ood_inputs).float()
output_tensor = torch.from_numpy(ood_targets).float()

ood_torch_dataset = MyDataset(input_tensor, output_tensor) 
ood_torch_dataset.scale_inputs(trained_scaler_inputs)
ood_torch_dataset.scale_outputs(trained_scaler_outputs)
ood_test_loader = torch.utils.data.DataLoader(ood_torch_dataset, batch_size = 1, shuffle = False)

#endregion


#region Create ML Model


#Hyperparamerets
learning_rate = 1e-4
epochs = 1000

#Network parameters
input_size = inputs.shape[1]
output_size = outputs.shape[1]
hidden_size = 120


#Model, Loss and Optimizer
model = TestModel(input_size, hidden_size, output_size)
loss_fn = nn.MSELoss() 
optimizer = Adam(model.parameters(), lr=learning_rate)

#endregion

#region Train model


filename = f"MinMax_lr_{learning_rate}_train_test_loss"
file = open(f"progress/{filename}.csv", "w")
file.write("Epoch,train_loss,val_loss,test_loss,diff_seed_loss,out_of_domain_loss \n")
file.close()


with tqdm(range(epochs + 1), desc='Epochs', unit='epoch', leave=True, bar_format='{desc}: {percentage:3.0f}%|{bar}|{postfix}') as t:
    best_val_loss = 5
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

        file = open(f"progress/{filename}.csv", "a")
        file.write(f"{epoch},{running_loss/len(train_loader)},{val_loss/len(val_loader)},{0},{0},{0} \n")
        file.close()

        if current_val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = current_val_loss
            train_loss_best_model = running_loss/len(train_loader) 
            save_filename = f"saved_models/best_{filename}.pth"
            torch.save(model, save_filename)

        if epoch % 100 == 0:

            print(f'Epoch {epoch+1}/{epochs} \nTraining Loss: {running_loss/len(train_loader):.4f} '
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

file = open(f"progress/{filename}.csv", "a")
file.write(f"{best_epoch},{train_loss_best_model},{best_val_loss},{test_loss/len(test_loader)},{diff_seed_loss/len(diff_seed_test_loader)},{ood_loss/len(ood_test_loader)} \n")
file.close()

#endregion


### TO DO
### Save model, using a validation set, to get the best model
### Include interventional data


#Now implement LIME


'''
def predict_lime(data):
    # Convert data into PyTorch tensors (if it is not already)
    data = torch.tensor(data, dtype=torch.float32)

    output = model(data)
    return output.detach().numpy()



explainer = lime_tabular.LimeTabularExplainer(test_loader.numpy(),   #may need to have this in the shape of a list or similar, not the entire dataloader
                                              feature_names=["y1", "y2"],
                                              mode = 'regression')





### Choose a sample from your test data that you want to explain.
i = 10    # index of the selected test data point
num_features = 5   #The top five contributing features
exp = explainer.explain_instance(test_data[i], predict_lime, num_features=5)

#Check your result. The result is shown as the influence of each feature on the prediction.
exp.show_in_notebook(show_table=True)

#Replace show_table with show_all=False to show less information.

#You can also print a list of tuples. Each tuple corresponds to a feature and how much it contributes to the predicted value.
print(exp.as_list())

'''