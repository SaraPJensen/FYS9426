def get_filename(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify):
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

    return filename


def get_model_name(Output_var, Deep, Scaling, Intervene, C_D, Independent, Simplify):

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

    model_name = f"saved_models/{Output_var}/best_{filename}.pth"

    return model_name