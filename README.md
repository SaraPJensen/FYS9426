# FYS9429 - Learning Stable Causal Explanations from Deep Neural Networks

This repository contains the all code for the project in FYS9429, spring 2024. The description will be limited to an overview of the high-level functionality of the different files, so no detailed description of the individual functions and classes within the files will be provided here. 

## Running code
None of the files take any command line arguments, so may simply be run as they are. Most of the functions have hard-coded the relevant variables, such as dataset types, output variable and network parameters. These are usually listed at the top of the file, so may be changed there. 
Some of the scripts may require the existence of certain folders in the directory, so these must be created if the code fails to run. 

## Code Overview
The code is separated into two main folders, one for the simple dataset and one for the complex, named **SCM_simple** and **SCM_complex**. Within these folders, the different results are stored in different, appropriately named folders, and all the runnable code is in the main folder.


### Simple model
The code for the simple dataset. The different python-files in the folder are as follows. 

#### Python scripts
- [scm_simple_dataset.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_simple/scm_simple_dataset.py): functions for generating all the different observational datasets.
- [scm_intv_simple_dataset.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_simple/scm_intv_simple_dataset.py): functions for generating all the different interventional datasets.
- [scm_simple_network.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_simple/scm_simple_network.py): script to train a model with the parameters listed at the beginning. Stores the progress and final results, as well as the best model. These are stored in the **progress** and **saved_models** folders respectively.
- [complete_summary.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_simple/complete_summary.py): script to generate a complete summary csv-file of the final results of the different models.
- [filename_funcs.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_simple/filename_funcs.py): helper functions to get the different filenames and model names.
- [shap_explain.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_simple/shap_explain.py): script to generate the \texttt{Shap} explanations of the model with parameters as listed at the beginning of the document. Results are stored in the **shap** folder. 
- [shap_summary.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_simple/shap_summary.py): script to calculate the explanation variance and loss, and summarize all the results from the Shapley explanations in one csv-file, stored in the **shap** folder.
- [plot_acc.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_simple/plot_acc.py): plotting code to plot the accuracy of the model with parameteres as listed at the beginning of the document.
- [plot_shap.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_simple/plot_shap.py): plotting code to plot the results from the variation and accuracy analysis of the Shapley explanations of the model with parameteres as listed at the beginning of the document.
- [correlate.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_simple/correlate.py): plotting code to calculate and plot the correlation matrix of the variation, explanation loss and model loss. 



### Complex model
The code for the complex dataset. The different python-files in the folder are as follows. 
All figures are stored in the **figures** folder. 

#### Python scripts
- [scm_complex_dataset.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_complex/scm_complex_dataset.py): functions for generating all the different observational datasets.
- [scm_intv_complex_dataset.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_complex/scm_intv_complex_dataset.py): functions for generating all the different interventional datasets.
- [scm_complex_network.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_complex/scm_complex_network.py): script to train a model with the parameters listed at the beginning. Stores the progress and final results, as well as the best model. These are stored in the **progress** and **saved_models** folders respectively.
- [complete_summary.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_complex/complete_summary.py): script to generate a complete summary csv-file of the final results of the different models.
- [filename_funcs.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_complex/filename_funcs.py): helper functions to get the different filenames and model names.
- [pysr_explain.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_complex/pysr_explain.py): script to generate the \texttt{PySR} explanations of the model with parameters as listed at the beginning of the document.
- [pysr_true.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_complex/pysr_true.py): script to generate the \texttt{PySR} explanations of true data generating function. Results are stored in the **pysr** folder. 
- [summary_pysr.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_complex/summary_pysr.py): script to calculate variance and accuracy of the PySR explanations, and summarize all the results in one csv-file, stored in the **pysr** folder. 
- [plot_acc.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_complex/plot_acc.py): plotting code to plot the accuracy of the model with parameteres as listed at the beginning of the document. 
- [plot_pysr.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_complex/plot_pysr.py): plotting code to plot the results from the variation and accuracy analysis of the PySR explanations of the model with parameteres as listed at the beginning of the document.
- [correlate.py](https://github.com/SaraPJensen/FYS9426/blob/main/SCM_complex/correlate.py): plotting code to calculate and plot the correlation matrix of the variation, explanation loss and model loss. 

