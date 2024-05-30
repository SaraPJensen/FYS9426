import pandas as pd
import numpy as np
from filename_funcs import get_filename, get_model_name
import torch
from scm_simple_network import MyDataset
from scm_simple_dataset import mixed_dataset_gen
from scm_intv_simple_dataset import scm_intv_dataset_gen
import sklearn

True_model = False

Scaling = False
Deep = False

Intervene = False 
C_D = False
Independent = False
Simplify = False

#Output_var = 'y1'
Output_var = 'y2'


#Define a dataset with observational data with an extended range
n_testing = 600

inputs, targets = mixed_dataset_gen(n_testing, seed = 54321)
input_tensor = torch.from_numpy(inputs).float()
output_tensor = torch.from_numpy(targets).float()
torch_dataset = MyDataset(input_tensor, output_tensor, Output_var, Simplify) 
unscaled_inputs, unscaled_targets = torch_dataset[:]



def make_modelpreds(a_coeff, b_coeff, c_coeff, d_coeff, e_coeff, X, targets):
        A = X[:, 0]
        B = X[:, 1]
        C = X[:, 2]
        D = X[:, 3]
        E = X[:, 4]

        pred = a_coeff*A + b_coeff*B + c_coeff*C + d_coeff*D + e_coeff*E
        loss = sklearn.metrics.mean_squared_error(targets, pred)

        return loss







def write_progress(source_file, save_filename, inputs, targets, Deep, Scaling, Intv, C_D, Independent, columns):
    if Scaling: 
        Scale_type = "MinMax"
    else:
        Scale_type = "Raw_data"

    df = pd.read_csv(f"{source_file}")
    avg_var = df.iat[-1,1]

    a_coeff = df["A"][6]
    b_coeff = df["B"][6]
    c_coeff = df["C"][6]
    d_coeff = df["D"][6]
    e_coeff = df["E"][6]

    avg_exp_loss = make_modelpreds(a_coeff, b_coeff, c_coeff, d_coeff, e_coeff, inputs, targets)

    acc_filename = f"progress/{Output_var}/{filename}.csv"
    df = pd.read_csv(acc_filename)
    losses = []
    for c in columns: 
        loss = df[c].iat[-1]
        losses.append(loss)
    avg_model_loss = np.mean(losses)

    save_file = open(save_filename, "a")
    save_file.write(f"{Deep},{Scale_type},{Intv},{C_D},{Independent},{avg_var},{a_coeff},{b_coeff},{c_coeff},{d_coeff},{e_coeff},{avg_exp_loss},{avg_model_loss}\n")  
    save_file.close()




save_filename = f"shap/{Output_var}/{Output_var}_summary.csv"
save_file = open(save_filename, "w")
save_file.write("Deep,Scaling,Intv,C_D,Independent,Avg_variance,avg_A,avg_B,avg_C,avg_D,avg_E,Avg_exp_loss,Avg_model_loss\n")
save_file.close()


columns = ["test_loss", "obsv_test_loss", "intv_test_loss", "out_of_domain_loss",  "diff_model_loss", "diff_mod_rand_loss"]

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)


Intervene = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)


C_D = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)

Intervene = False
C_D = False
Independent = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)



Deep = False
Scaling = True

Intervene = False
C_D = False
Independent = False
Simplify = False

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)

Intervene = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)

C_D = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)

Intervene = False
C_D = False
Independent = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)



Deep = True
Scaling = False

Intervene = False
C_D = False
Independent = False
Simplify = False

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)

Intervene = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)

C_D = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)

Intervene = False
C_D = False
Independent = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)


Deep = True
Scaling = True

Intervene = False
C_D = False
Independent = False
Simplify = False

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)

Intervene = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)

C_D = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)

Intervene = False
C_D = False
Independent = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, unscaled_inputs, unscaled_targets, Deep, Scaling, Intervene, C_D, Independent, columns)





