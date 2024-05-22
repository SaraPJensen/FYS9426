import pandas as pd

from filename_funcs import get_filename, get_model_name



def write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intv, C_D, Independent):
    if Scaling: 
        Scale_type = "MinMax"
    else:
        Scale_type = "Raw_data"

    df = pd.read_csv(f"{source_file}")
    avg_var = df.iat[-1,1] #.values

    save_file = open(save_filename, "a")
    save_file.write(f"{Deep},{Scale_type},{Intv},{C_D},{Independent},{avg_var}\n")  
    save_file.close()


True_model = False

Scaling = False
Deep = False

Intervene = False 
C_D = False
Independent = False
Simplify = False

Output_var = 'y1'
#Output_var = 'y2'



save_filename = f"shap/{Output_var}/{Output_var}_summary.csv"
save_file = open(save_filename, "w")
save_file.write("Deep,Scaling,Intv,C_D,Independent,Avg_variance\n")
save_file.close()


filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)


Intervene = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)


C_D = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)

Intervene = False
C_D = False
Independent = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



Deep = False
Scaling = True

Intervene = False
C_D = False
Independent = False
Simplify = False

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)

Intervene = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)

C_D = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)

Intervene = False
C_D = False
Independent = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)



Deep = True
Scaling = False

Intervene = False
C_D = False
Independent = False
Simplify = False

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)

Intervene = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)

C_D = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)

Intervene = False
C_D = False
Independent = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)


Deep = True
Scaling = True

Intervene = False
C_D = False
Independent = False
Simplify = False

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)

Intervene = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)

C_D = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)

Intervene = False
C_D = False
Independent = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
source_file = f"shap/{Output_var}/{filename}.csv"
write_progress(source_file, save_filename, Output_var, Deep, Scaling, Intervene, C_D, Independent)





