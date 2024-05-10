import numpy as np
import pandas as pd


def progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify):
    file_intv = ''
    if Intervene:
        if C_D:
            file_intv = "Intv_C_D_"
        else:
            file_intv = "Intv_"


    if Scaling:
        filename = f"{Output_var}_{file_intv}MinMax_lr_0.001"
    else:
        filename = f"{Output_var}_{file_intv}Raw_data_lr_0.001"

    if Independent:
        filename = "indep_" + filename

    if Simplify: 
        filename = "simple_" + filename

    if Deep:
        filename = "deeeeep_" + filename

    filename = f"progress/{Output_var}/" + filename

    return filename


def write_progress(filename, save_file, Output_var, Deep, Scaling, Intv, C_D, Independent, Simple):
    if Scaling: 
        Scale_type = "MinMax"
    else:
        Scale_type = "Raw_data"
    df = pd.read_csv(f"{filename}.csv")
    # print(filename)
    # print(df)
    last_row = df.iloc[-1].values
    # print()
    # print(last_row)
    
    #save_file.write("Deep,Scaling,Rescaled,Intv,C_D,train_loss,val_loss,test_loss,diff_seed_loss,out_of_domain_loss,diff_model_loss,diff_mod_rand_loss,obsv_test_loss,intv_test_loss \n")
    
    save_file.write(f"{Deep},{Scale_type},{Intv},{C_D},{Independent},{Simple},{last_row[1]},{last_row[2]},{last_row[3]},{last_row[4]},{last_row[5]},{last_row[6]},{last_row[7]},{last_row[8]},{last_row[9]},{last_row[10]},{last_row[0]}\n")  
    #input()



learning_rate = 0.001
#Output_var = 'y1'
Output_var = 'y2'

save_filename = f"progress/{Output_var}/{Output_var}_summary.csv"
save_file = open(f"{save_filename}", "w")
save_file.write("Deep,Scaling,Intv,C_D,Independent,Simple,train_loss,val_loss,test_loss,diff_seed_loss,out_of_domain_loss,diff_model_loss,diff_mod_rand_loss,obsv_test_loss,intv_test_loss,obsv_normal_loss,best_epoch \n")


Deep = False
Scaling = False

Intervene = False
C_D = False
Independent = False
Simplify = False



filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Intervene = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

C_D = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Intervene = False
C_D = False
Independent = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Simplify = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Independent = False
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)



Deep = False
Scaling = True

Intervene = False
C_D = False
Independent = False
Simplify = False


filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Intervene = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

C_D = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Intervene = False
C_D = False
Independent = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Simplify = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Independent = False
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)



Deep = True
Scaling = False

Intervene = False
C_D = False
Independent = False
Simplify = False


filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Intervene = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

C_D = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Intervene = False
C_D = False
Independent = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Simplify = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Independent = False
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)


Deep = True
Scaling = True

Intervene = False
C_D = False
Independent = False
Simplify = False


filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Intervene = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

C_D = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Intervene = False
C_D = False
Independent = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Simplify = True
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)

Independent = False
filename = progress_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
write_progress(filename, save_file, Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)


