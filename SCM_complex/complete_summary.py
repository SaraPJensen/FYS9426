import numpy as np
import pandas as pd


def progress_filename(Output_var, Deep, Intv, C_D, Rescale, Input_scaling, Output_scaling, lr):
    file_intv = ''
    if Intv:
        if C_D:
            file_intv = "Intv_C_D_noInfo_"
        else:
            file_intv = "Intv_noInfo_"


    if Input_scaling and Output_scaling:
        filename = f"{Output_var}_{file_intv}MinMax_all_lr_{lr}"
    elif Input_scaling:
        filename = f"{Output_var}_{file_intv}MinMax_inputs_raw_outputs_lr_{lr}"
    elif Output_scaling:
        filename = f"{Output_var}_{file_intv}MinMax_outputs_raw_inputs_lr_{lr}"
    else:
        filename = f"{Output_var}_{file_intv}Raw_data_lr_{lr}"

    if Deep:
        filename = "deeeeep_" + filename

    if Rescale: 
        add = "rescale_" 
    else: 
        add = ''

    filename = f"progress/{Output_var}/test_model/{add}{filename}"

    return filename


def write_progress(filename, save_file, Output_var, Deep, Intv, C_D, Rescale, Input_scaling, Output_scaling, lr):
    if Input_scaling: 
        Scaling = "MinMax"
    else:
        Scaling = "Raw_data"
    df = pd.read_csv(f"{filename}.csv")
    print(filename)
    print(df)
    last_row = df.iloc[-1].values/100
    print()
    print(last_row)
    
    #save_file.write("Deep,Scaling,Rescaled,Intv,C_D,train_loss,val_loss,test_loss,diff_seed_loss,out_of_domain_loss,diff_model_loss,diff_mod_rand_loss,obsv_test_loss,intv_test_loss \n")
    
    save_file.write(f"{Deep},{Scaling},{Rescale},{Intv},{C_D},{last_row[0]},{last_row[1]},{last_row[2]},{last_row[3]},{last_row[4]},{last_row[5]},{last_row[6]},{last_row[7]},{last_row[8]}\n")  
    input()



lr = 0.001
#Output_var = 'y1'
Output_var = 'y2'

save_filename = f"progress/{Output_var}/test_model/{Output_var}_test_summary.csv"
save_file = open(f"{save_filename}", "w")
save_file.write("Deep,Scaling,Rescaled,Intv,C_D,train_loss,val_loss,test_loss,diff_seed_loss,out_of_domain_loss,diff_model_loss,diff_mod_rand_loss,obsv_test_loss,intv_test_loss \n")


Input_scaling = False
Output_scaling = False
Intervene = False
C_D = False
Deep = False
Rescale = False



filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

Intervene = True
filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

C_D = True
filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)



Input_scaling = True
Output_scaling = True
# Intervene = False
# C_D = False
# filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
# write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

# Intervene = True
# filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
# write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

# C_D = True
# filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
# write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)



Rescale = True
Intervene = False
C_D = False
filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

Intervene = True
filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

C_D = True
filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)


Input_scaling = False
Output_scaling = False
Intervene = False
C_D = False
Deep = True
Rescale = False

filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

Intervene = True
filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

C_D = True
filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)



Input_scaling = True
Output_scaling = True
# Intervene = False
# C_D = False
# filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
# write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

# Intervene = True
# filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
# write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

# C_D = True
# filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
# write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)



Rescale = True
Intervene = False
C_D = False
filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

Intervene = True
filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)

C_D = True
filename = progress_filename(Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)
write_progress(filename, save_file, Output_var, Deep, Intervene, C_D, Rescale, Input_scaling, Output_scaling, lr)







