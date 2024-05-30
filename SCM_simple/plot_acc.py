import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from filename_funcs import get_filename, get_model_name
from pastamarkers import markers as pasta
from matplotlib import rc


Scaling = False
Deep = True


#Output_var = 'y1'
Output_var = 'y2'

if Scaling: 
    tit_scale = "min-max scaling"
else: 
    tit_scale = "no scaling"

if Deep: 
    tit_depth = "deep network"
else: 
    tit_depth = "shallow network"


models = ["Obsv", "Intv", "C_D", "Indp", "Simple"]
results = []

if Output_var == 'y1':
    pasta_types = [pasta.soli, pasta.pipe, pasta.gramigna, pasta.fiori, pasta.radiatori]
else:
    pasta_types = [pasta.tagliatelle, pasta.penne, pasta.farfalline, pasta.spighe, pasta.conchiglie]


colours = ["firebrick", "goldenrod", "forestgreen", "lightseagreen", "steelblue"]
columns = ["test_loss", "obsv_test_loss", "intv_test_loss", "out_of_domain_loss",  "diff_model_loss", "diff_mod_rand_loss"]
column_names = ["Test set", "Obsv", "Intv", "OOD",  "Diff. mod.", "Rand. mod."]



Intervene = False 
C_D = False
Independent = False
Simplify = False

filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
acc_filename = f"progress/{Output_var}/{filename}.csv"
df = pd.read_csv(acc_filename)
losses = []
for c in columns: 
    loss = df[c].iat[-1]
    losses.append(loss)
results.append(losses)


Intervene = True 
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
acc_filename = f"progress/{Output_var}/{filename}.csv"
df = pd.read_csv(acc_filename)
losses = []
for c in columns: 
    loss = df[c].iat[-1]
    losses.append(loss)
results.append(losses)


C_D = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
acc_filename = f"progress/{Output_var}/{filename}.csv"
df = pd.read_csv(acc_filename)
losses = []
for c in columns: 
    loss = df[c].iat[-1]
    losses.append(loss)
results.append(losses)

Intervene = False
C_D = False
Independent = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
acc_filename = f"progress/{Output_var}/{filename}.csv"
df = pd.read_csv(acc_filename)
losses = []
for c in columns: 
    loss = df[c].iat[-1]
    losses.append(loss)
results.append(losses)


Independent = False
Simplify = True
filename = get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify)
acc_filename = f"progress/{Output_var}/{filename}.csv"
df = pd.read_csv(acc_filename)
losses = []
for c in columns: 
    loss = df[c].iat[-1]
    losses.append(loss)
results.append(losses)


rc('font',**{'family':'sans-serif','sans-serif':['Avenir']})
rc('text', usetex=True)


x = column_names

for y_vals, model_label, pasta_type, colour in zip(results, models, pasta_types, colours):
    plt.plot(x, 
            np.log(y_vals), 
            label = model_label, 
            marker = pasta_type, 
            color = colour,
            markersize = 18, 
            linestyle = '') 

plt.legend(title="ML model", fancybox=True, title_fontsize = 14, fontsize = 13)
plt.xticks(rotation=30, ha="right", fontsize = 13)
plt.yticks(fontsize=13)
plt.title(f"Test loss, simple model, {tit_scale}, {tit_depth}, {Output_var}", fontsize=17)#, **hfont)
plt.ylabel("Avg. MSE loss, logarithmic", fontsize=16)
plt.xlabel("Test dataset", fontsize=16)
plt.tight_layout()


plt.savefig(f"figures/loss_{Output_var}_{tit_scale}_{tit_depth}.png", dpi = 300)
#plt.show()

