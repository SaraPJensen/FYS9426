import torch 
import torch.nn as nn
import numpy as np
from scm_simple_dataset import scm_dataset_gen, scm_out_of_domain, scm_diff_seed
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.init as init
from lime import lime_tabular

torch.manual_seed(2)
np.random.seed(2)




#region Dataset functions

def normalize_data(inputs, targets):
    # Flatten the input and target tensors
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # Calculate mean and standard deviation for normalization
    input_mean = inputs.mean(dim=0)
    input_std = inputs.std(dim=0)
    target_mean = targets.mean(dim=0)
    target_std = targets.std(dim=0)


    # Normalize the data
    inputs_normalized = (inputs - input_mean) / input_std
    targets_normalized = (targets - target_mean) / target_std

    return inputs_normalized, targets_normalized




class MyDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

        #self.inputs, self.targets = normalize_data(inputs, targets)
    
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

inputs, outputs = scm_dataset_gen(n_datapoints)

batch_size = 32


#Make torch dataset
input_tensor = torch.from_numpy(inputs).float()
output_tensor = torch.from_numpy(outputs).float()

torch_dataset = MyDataset(input_tensor, output_tensor) 
train_data, test_data = torch.utils.data.random_split(torch_dataset, [int(0.7*n_datapoints), int(0.3*n_datapoints)])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)


### Define a dataset with different seed
n_diff_seed = 500
diff_seed_inputs, diff_seed_targets = scm_diff_seed(n_diff_seed)

diff_input_tensor = torch.from_numpy(diff_seed_inputs).float()
diff_output_tensor = torch.from_numpy(diff_seed_targets).float()

diff_seed_torch_dataset = MyDataset(diff_input_tensor, diff_output_tensor) 
diff_seed_test_loader = torch.utils.data.DataLoader(diff_seed_torch_dataset, batch_size = 1, shuffle = False)


### Define an out-of-domain dataset
n_ood = 500
ood_inputs, ood_targets = scm_out_of_domain(n_ood)

input_tensor = torch.from_numpy(ood_inputs).float()
output_tensor = torch.from_numpy(ood_targets).float()

ood_torch_dataset = MyDataset(input_tensor, output_tensor) 
ood_test_loader = torch.utils.data.DataLoader(ood_torch_dataset, batch_size = 1, shuffle = False)

#endregion


#region Create ML Model


#Hyperparamerets
learning_rate = 1e-3
epochs = 500

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

with tqdm(range(epochs + 1), desc='Epochs', unit='epoch', leave=True, bar_format='{desc}: {percentage:3.0f}%|{bar}|{postfix}') as t:
    for epoch in t:

        model.train()
        running_loss = 0 
        for inputs, targets in train_loader:
            # Forward pass
            pred = model(inputs) 
            loss = loss_fn(pred, targets) 


            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*100

            optimizer.zero_grad()

        t.set_postfix(Loss=f'{(running_loss/len(train_loader)):.6f}') 
        t.refresh() # Refresh to show postfix


        if epoch % 100 == 0:
            model.eval()
            test_loss = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    pred = model(inputs) 
                    # print(pred)
                    # input()
                    loss = loss_fn(pred, targets) 
                    test_loss += loss.item()*100

            print(f'Epoch {epoch+1}/{epochs} \nTraining Loss: {running_loss/len(train_loader):.4f} '
            f'\nTest Loss: {test_loss/len(test_loader):.6f}')
            print()

print()
print("Finished training")
print(f'Final test Loss: {test_loss/len(test_loader):.6f}')

#endregion


#region Test with other data
print()
print("Test with different seed")
model.eval()
diff_seed_loss = 0

with torch.no_grad():
    for inputs, targets in diff_seed_test_loader:
        pred = model(inputs)
        loss = loss_fn(pred, targets) 
        diff_seed_loss += loss.item()*100

print(f'Different seed Loss: {diff_seed_loss/len(diff_seed_test_loader):.6f}')
print()



print("Test with out-of-domain data")

model.eval()
ood_loss = 0

with torch.no_grad():
    for inputs, targets in ood_test_loader:

        pred = model(inputs)
        loss = loss_fn(pred, targets) 
        ood_loss += loss.item()*100

print(f'Out of domain Loss: {ood_loss/len(ood_test_loader):.6f}')
print()


#endregion


### TO DO
### Save model
### Save training and test loss
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