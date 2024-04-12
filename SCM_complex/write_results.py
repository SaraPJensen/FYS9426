import numpy as np
import pandas as pd


def progress_filename(Intv, Intv_info, C_D, Input_scaling, Output_scaling, lr):
    file_intv = ''
    if Intv:
        if C_D:
            if Intv_info: 
                file_intv = "Intv_C_D_Info_"
            else:
                file_intv = "Intv_C_D_noInfo_"

        else:
            if Intv_info:
                file_intv = "Intv_Info_"
            else:
                file_intv = "Intv_noInfo_"


    if Input_scaling and Output_scaling:
        filename = f"{output_var}_{file_intv}MinMax_all_lr_{lr}"
    elif Input_scaling:
        filename = f"{output_var}_{file_intv}MinMax_inputs_raw_outputs_lr_{lr}"
    elif Output_scaling:
        filename = f"{output_var}_{file_intv}MinMax_outputs_raw_inputs_lr_{lr}"
    else:
        filename = f"{output_var}_{file_intv}Raw_data_lr_{lr}"


    if Input_scaling and Output_scaling:
        scaling_info = "MinMax_all"
    elif Input_scaling: 
        scaling_info = "MinMax_inputs_raw_outputs"
    elif Output_scaling:
        scaling_info = "MinMax_outputs_raw_inputs"
    else:
        scaling_info = "Raw_data"

    return filename, scaling_info


def write_progress(filename, scaling_info, save_file):
    df = pd.read_csv(f"progress/{filename}.csv")
    last_row = df.iloc[-1].values
    save_file.write(f"{Intv},{Intv_info},{C_D},{scaling_info},{lr},{last_row[0]},{last_row[1]},{last_row[2]},{last_row[3]},{last_row[4]},{last_row[5]},{last_row[6]},{last_row[7]},{last_row[8]},{last_row[9]}\n")  



#output_var = "y1"
output_var = "y2"

save_filename = f"{output_var}_summary.csv"
save_file = open(f"progress/{save_filename}", "w")
save_file.write("Intv,Intv_info,C_D,Scaling,lr,best_epoch,train_loss,val_loss,test_loss,diff_seed_loss,out_of_domain_loss,diff_model_loss,diff_mod_rand_loss,obsv_test_loss,intv_test_loss \n")


Intv = False
Intv_info = False
C_D = False
Input_scaling = False
Output_scaling = False
lr = 0.001

filename, scaling_info = progress_filename(Intv, Intv_info, C_D, Input_scaling, Output_scaling, lr)
write_progress(filename, scaling_info, save_file)


Intv = True
filename, scaling_info = progress_filename(Intv, Intv_info, C_D, Input_scaling, Output_scaling, lr)
write_progress(filename, scaling_info, save_file)


C_D = True
filename, scaling_info = progress_filename(Intv, Intv_info, C_D, Input_scaling, Output_scaling, lr)
write_progress(filename, scaling_info, save_file)


Input_scaling = True
Output_scaling = True
Intv = False
Intv_info = False
C_D = False

filename, scaling_info = progress_filename(Intv, Intv_info, C_D, Input_scaling, Output_scaling, lr)
write_progress(filename, scaling_info, save_file)

Intv = True
filename, scaling_info = progress_filename(Intv, Intv_info, C_D, Input_scaling, Output_scaling, lr)
write_progress(filename, scaling_info, save_file)

C_D = True
filename, scaling_info = progress_filename(Intv, Intv_info, C_D, Input_scaling, Output_scaling, lr)
write_progress(filename, scaling_info, save_file)