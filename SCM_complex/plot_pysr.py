import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from filename_funcs import get_filename, get_model_name
from pastamarkers import markers as pasta
from matplotlib import rc



Output_var = 'y1'
#Output_var = 'y2'

no_ood = True

Variance = True
Exp_loss = False
Model_loss = False

if Variance:
    dep_var = "variance"
    tit_var = "Variance of explanations"
if Exp_loss:
    dep_var = "explanation loss"
    tit_var = "Average explanation loss"
if Model_loss:
    dep_var = "model loss"
    tit_var = "Average model loss"

if no_ood:
    tit_var = tit_var + ", no ood" 

if no_ood:
    filename = f"pysr/{Output_var}/{Output_var}_no_ood_summary.csv"
else:
    filename = f"pysr/{Output_var}/{Output_var}_summary.csv"


df = pd.read_csv(filename)


if Variance:
    variances = df["Avg_variance"].to_list()
    results = []
    for i in range(0, 4):
        results.append([variances[p+i] for p in range(0, len(variances), 4)])

elif Exp_loss:
    exp_loss = df["Avg_exp_loss"].to_list()
    results = []
    for i in range(0, 4):
        results.append([exp_loss[p+i] for p in range(0, len(exp_loss), 4)])

else:
    model_loss = df["Avg_model_loss"].to_list()
    results = []
    for i in range(0, 4):
        results.append([model_loss[p+i] for p in range(0, len(model_loss), 4)])



rc('font',**{'family':'sans-serif','sans-serif':['Avenir']})
rc('text', usetex=True)



networks = ["Shallow, raw data", "Shallow, min-max", "Deep, raw data", "Deep, min-max"]  #These are the different lines to plot

datasets = ["Obsv", "Intv", "C_D", "Indp"]


if Output_var == 'y1':
    pasta_types = [pasta.ruote, pasta.maccheroni, pasta.gnocchi, pasta.farfalle, pasta.ravioli]
else:
    pasta_types = [pasta.creste, pasta.tortellini, pasta.cavatappi, pasta.rigatoni, pasta.stelline]


colours = ["firebrick", "goldenrod", "forestgreen", "lightseagreen", "steelblue"]



for y_vals, model_label, pasta_type, colour in zip(results, datasets, pasta_types, colours):
    plt.plot(networks, 
            np.log(y_vals), 
            label = model_label, 
            marker = pasta_type, 
            color = colour,
            markersize = 20, 
            linestyle = '') 

#plt.ylim(np.min(np.log(results))-0.5, 7)
plt.legend(title="ML model", fancybox=True, title_fontsize = 13, fontsize = 11)
plt.xticks(rotation=30, ha="right", fontsize = 13)
plt.yticks(fontsize=13)
plt.title(f"{tit_var}, complex model, {Output_var}", fontsize=18)#, **hfont)
plt.ylabel(f"Avg. {dep_var}, logarithmic", fontsize=16)
plt.xlabel("Network, scaling", fontsize=16)
plt.tight_layout()


if no_ood:
    plt.savefig(f"figures/{dep_var}_{Output_var}_no_ood.png", dpi = 300)

else:
    plt.savefig(f"figures/{dep_var}_{Output_var}.png", dpi = 300)
#plt.show()


