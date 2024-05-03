import numpy as np 



def datapoint_gen(a, b, e):
    d = 2*b 
    c = 5*a - a/d
    y1 = 3*a**2 + d**3 + a*d - d**2
    y2 = - d**2 + 4*d + np.sqrt(e)

    return c, d, y1, y2




def intv_a(sub_n_datapoints, intv_info = False):

    A = np.random.uniform(-2, 4, size=sub_n_datapoints)
    B = np.random.uniform(3, 5.5, size=sub_n_datapoints)
    E = np.random.uniform(0, 8, size=sub_n_datapoints)

    D = 2*B 
    C = 5*A - A/D
    Y1 = 3*A**2 + D**3 + A*D - D**2
    Y2 = - D**2 + 4*D + np.sqrt(E)


    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([1, 0, 0, 0, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))


    return inputs, outputs


def intv_b(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-2, 4, size=sub_n_datapoints)
    B = np.random.uniform(3, 5.5, size=sub_n_datapoints)
    E = np.random.uniform(0, 8, size=sub_n_datapoints)

    D = 2*B 
    C = 5*A - A/D
    Y1 = 3*A**2 + D**3 + A*D - D**2
    Y2 = - D**2 + 4*D + np.sqrt(E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([0, 1, 0, 0, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))

    return inputs, outputs


def intv_c(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-2, 4, size=sub_n_datapoints)
    B = np.random.uniform(3, 5.5, size=sub_n_datapoints)
    E = np.random.uniform(0, 8, size=sub_n_datapoints)

    C = np.random.uniform(-9.8, 19.6, size=sub_n_datapoints)  #Intervention
    D = 2*B 
    Y1 = 3*A**2 + D**3 + A*D - D**2
    Y2 = - D**2 + 4*D + np.sqrt(E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([0, 0, 1, 0, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))

    return inputs, outputs


def intv_d(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-2, 4, size=sub_n_datapoints)
    B = np.random.uniform(3, 5.5, size=sub_n_datapoints)
    E = np.random.uniform(0, 8, size=sub_n_datapoints)

    D = np.random.uniform(6, 11, size=sub_n_datapoints)   #Interventions
    C = 5*A - A/D
    Y1 = 3*A**2 + D**3 + A*D - D**2
    Y2 = - D**2 + 4*D + np.sqrt(E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([0, 0, 0, 1, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))

    return inputs, outputs

def intv_e(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-2, 4, size=sub_n_datapoints)
    B = np.random.uniform(3, 5.5, size=sub_n_datapoints)
    E = np.random.uniform(0, 8, size=sub_n_datapoints)

    D = 2*B 
    C = 5*A - A/D
    Y1 = 3*A**2 + D**3 + A*D - D**2
    Y2 = - D**2 + 4*D + np.sqrt(E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([0, 0, 0, 0, 1])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))

    return inputs, outputs





def scm_intv_dataset_gen(n_datapoints, intv_info = False, seed = 5):
    np.random.seed(seed)

    sub_n_datapoints = int(n_datapoints/5)

    I_A_in, I_A_out = intv_a(sub_n_datapoints, intv_info)
    I_B_in, I_B_out = intv_b(sub_n_datapoints, intv_info)
    I_C_in, I_C_out = intv_c(sub_n_datapoints, intv_info)
    I_D_in, I_D_out = intv_d(sub_n_datapoints, intv_info)
    I_E_in, I_E_out = intv_e(sub_n_datapoints, intv_info)

    inputs = np.row_stack((I_A_in, I_B_in, I_C_in, I_D_in, I_E_in))
    outputs = np.row_stack((I_A_out, I_B_out, I_C_out, I_D_out, I_E_out))

    return inputs, outputs




def scm_intv_c_d_dataset_gen(n_datapoints, intv_info = False, seed = 5):
    np.random.seed(seed)

    sub_n_datapoints = int(n_datapoints/2)

    I_C_in, I_C_out = intv_c(sub_n_datapoints, intv_info)
    I_D_in, I_D_out = intv_d(sub_n_datapoints, intv_info)

    inputs = np.row_stack((I_C_in, I_D_in))
    outputs = np.row_stack((I_C_out, I_D_out))

    return inputs, outputs






def intv_a_ood(n_out_of_domain, intv_info = False):
    A = np.random.uniform(-3, 5, size=n_out_of_domain)
    B = np.random.uniform(0.5, 4.5, size=n_out_of_domain)
    E = np.random.uniform(2, 7, size=n_out_of_domain)

    D = 2*B 
    C = 5*A - A/D
    Y1 = 3*A**2 + D**3 + A*D - D**2
    Y2 = - D**2 + 4*D + np.sqrt(E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs


def intv_b_ood(n_out_of_domain, intv_info = False):
    A = np.random.uniform(-3, 5, size=n_out_of_domain)
    B = np.random.uniform(0.5, 4.5, size=n_out_of_domain)
    E = np.random.uniform(2, 7, size=n_out_of_domain)

    D = 2*B 
    C = 5*A - A/D
    Y1 = 3*A**2 + D**3 + A*D - D**2
    Y2 = - D**2 + 4*D + np.sqrt(E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs


def intv_c_ood(n_out_of_domain, intv_info = False):
    A = np.random.uniform(-3, 5, size=n_out_of_domain)
    B = np.random.uniform(0.5, 4.5, size=n_out_of_domain)
    E = np.random.uniform(2, 7, size=n_out_of_domain)

    C = np.random.uniform(-14.6, 24.4, size=n_out_of_domain)   
    D = 2*B 
    Y1 = 3*A**2 + D**3 + A*D - D**2
    Y2 = - D**2 + 4*D + np.sqrt(E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs


def intv_d_ood(n_out_of_domain, intv_info = False):
    A = np.random.uniform(-3, 5, size=n_out_of_domain)
    B = np.random.uniform(0.5, 4.5, size=n_out_of_domain)
    E = np.random.uniform(2, 7, size=n_out_of_domain)

    D = np.random.uniform(1, 9, size=n_out_of_domain)   
    C = 5*A - A/D
    Y1 = 3*A**2 + D**3 + A*D - D**2
    Y2 = - D**2 + 4*D + np.sqrt(E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs


def intv_e_ood(n_out_of_domain, intv_info = False):
    A = np.random.uniform(-3, 5, size=n_out_of_domain)
    B = np.random.uniform(0.5, 4.5, size=n_out_of_domain)
    E = np.random.uniform(2, 7, size=n_out_of_domain)

    D = 2*B 
    C = 5*A - A/D
    Y1 = 3*A**2 + D**3 + A*D - D**2
    Y2 = - D**2 + 4*D + np.sqrt(E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs





def scm_intv_ood(n_datapoints, intv_info = False, seed = 5):
    np.random.seed(seed)

    sub_n_datapoints = int(n_datapoints/5)

    I_A_in, I_A_out = intv_a_ood(sub_n_datapoints, intv_info)
    I_B_in, I_B_out = intv_b_ood(sub_n_datapoints, intv_info)
    I_C_in, I_C_out = intv_c_ood(sub_n_datapoints, intv_info)
    I_D_in, I_D_out = intv_d_ood(sub_n_datapoints, intv_info)
    I_E_in, I_E_out = intv_e_ood(sub_n_datapoints, intv_info)

    inputs = np.row_stack((I_A_in, I_B_in, I_C_in, I_D_in, I_E_in))
    outputs = np.row_stack((I_A_out, I_B_out, I_C_out, I_D_out, I_E_out))

    return inputs, outputs
    



