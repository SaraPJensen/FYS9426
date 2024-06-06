import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from filename_funcs import get_filename, get_model_name
from pastamarkers import markers as pasta
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


Output_var = 'y1'
filename = f"shap/{Output_var}/{Output_var}_summary.csv"


df = pd.read_csv(filename)


coeff_variances_y1 = df["Avg_coeff_variance"].to_list()
pred_variance_y1 = df["Avg_pred_variance"].to_list()
exp_loss_y1 = df["Avg_exp_loss"].to_list()
model_loss_y1 = df["Avg_model_loss"].to_list()



Output_var = 'y2'

filename = f"shap/{Output_var}/{Output_var}_summary.csv"


df = pd.read_csv(filename)

coeff_variances_y2 = df["Avg_coeff_variance"].to_list()
pred_variance_y2 = df["Avg_pred_variance"].to_list()
exp_loss_y2 = df["Avg_exp_loss"].to_list()
model_loss_y2 = df["Avg_model_loss"].to_list()


all_coeff_variances = coeff_variances_y1 + coeff_variances_y2
all_pred_variances = pred_variance_y1 + pred_variance_y2
all_exp_loss = exp_loss_y1 + exp_loss_y2
all_model_loss = model_loss_y1 + model_loss_y2


combined = np.stack((all_coeff_variances, all_pred_variances, all_exp_loss, all_model_loss), axis = 0)

combined_y1 = np.stack((coeff_variances_y1, exp_loss_y1, model_loss_y1), axis = 0)


correlation_matrix = np.corrcoef(combined)

print(correlation_matrix)


# Create the heatmap with diagonal orientation
plt.figure(figsize=(8, 7.2))
heatmap = plt.pcolor(correlation_matrix[::-1, :], cmap='winter')

# Add the correlation coefficients to the heatmap
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        text = plt.text(j + 0.5, i + 0.5, '{:.2f}'.format(correlation_matrix[::-1, :][i, j]), ha='center', va='center', color='black', fontsize = 16)
        # Add a white outline to the numbers
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])

# Add a colorbar
colorbar = plt.colorbar(heatmap)
colorbar.ax.tick_params(labelsize=12)  # Set the fontsize of the colorbar ticks


rc('font',**{'family':'sans-serif','sans-serif':['Avenir']})
rc('text', usetex=True)

# Set the labels with angled tick labels
tick_labels = ['Avg. coeff. variance', 'Avg. exp. pred. variance', 'Avg. explanation loss', 'Avg. model loss']
plt.xticks(np.arange(0.5, 4.5), tick_labels, rotation=45, ha='right', fontsize = 13)
plt.yticks(np.arange(0.5, 4.5), tick_labels[::-1], rotation=45, ha='right', fontsize = 13)

# Add a title
plt.title('Correlation matrix for variance and loss of simple datasets\n', fontsize = 20)

plt.tight_layout()
plt.savefig("figures/correlation_mat_simple.png", dpi = 300)
# Show the plot
plt.show()

