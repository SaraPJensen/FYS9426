import numpy as np 



def datapoint_gen(a, b, e):
    d = 2*b 
    c = 5*a + d
    y1 = 3.5*a + 0.5*d 
    y2 = - 2*d + 0.2*e

    return c, d, y1, y2




def intv_a(sub_n_datapoints, intv_info = False):

    A = np.random.uniform(-5, 10, size=sub_n_datapoints)
    B = np.random.uniform(5, 20, size=sub_n_datapoints)
    E = np.random.uniform(0, 15, size=sub_n_datapoints)

    D = 2*B 
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = - 2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([1, 0, 0, 0, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))


    return inputs, outputs


def intv_b(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-5, 10, size=sub_n_datapoints)
    B = np.random.uniform(5, 20, size=sub_n_datapoints)
    E = np.random.uniform(0, 15, size=sub_n_datapoints)

    D = 2*B 
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = - 2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([0, 1, 0, 0, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))

    return inputs, outputs


def intv_c(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-5, 10, size=sub_n_datapoints)
    B = np.random.uniform(5, 20, size=sub_n_datapoints)
    E = np.random.uniform(0, 15, size=sub_n_datapoints)

    C = np.random.uniform(-15, 90, size=sub_n_datapoints)  #Intervention
    D = 2*B   #Max 40, min 10
    Y1 = 3.5*A + 0.5*D 
    Y2 = - 2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([0, 0, 1, 0, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))

    return inputs, outputs


def intv_d(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-5, 10, size=sub_n_datapoints)
    B = np.random.uniform(5, 20, size=sub_n_datapoints)
    E = np.random.uniform(0, 15, size=sub_n_datapoints)

    D = np.random.uniform(10, 40, size=sub_n_datapoints)   #Interventions
    C = 5*A + D  #Max 50+40=90, min. -25+10=-15
    Y1 = 3.5*A + 0.5*D 
    Y2 = - 2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([0, 0, 0, 1, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))

    return inputs, outputs

def intv_e(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-5, 10, size=sub_n_datapoints)
    B = np.random.uniform(5, 20, size=sub_n_datapoints)
    E = np.random.uniform(0, 15, size=sub_n_datapoints)

    D = 2*B 
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = - 2*D + 0.2*E

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
    #return shuffled_inputs, shuffled_outputs 


def scm_intv_c_d_dataset_gen(n_datapoints, intv_info = False, seed = 5):
    np.random.seed(seed)

    sub_n_datapoints = int(n_datapoints/2)

    I_C_in, I_C_out = intv_c(sub_n_datapoints, intv_info)
    I_D_in, I_D_out = intv_d(sub_n_datapoints, intv_info)

    inputs = np.row_stack((I_C_in, I_D_in))
    outputs = np.row_stack((I_C_out, I_D_out))

    return inputs, outputs
    #return shuffled_inputs, shuffled_outputs 




def scm_intv_diff_seed(n_diff_seed, intv_info = False):
    inputs, outputs = scm_intv_dataset_gen(n_diff_seed, intv_info, seed = 12345)

    return inputs, outputs 









def intv_a_ood(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-3, 12, size=sub_n_datapoints)
    B = np.random.uniform(20, 22, size=sub_n_datapoints)
    E = np.random.uniform(2, 8, size=sub_n_datapoints)

    D = 2*B 
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = - 2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([1, 0, 0, 0, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))


    return inputs, outputs


def intv_b_ood(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-3, 12, size=sub_n_datapoints)
    B = np.random.uniform(20, 22, size=sub_n_datapoints)
    E = np.random.uniform(2, 8, size=sub_n_datapoints)

    D = 2*B 
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = - 2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([0, 1, 0, 0, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))

    return inputs, outputs


def intv_c_ood(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-3, 12, size=sub_n_datapoints)
    B = np.random.uniform(20, 22, size=sub_n_datapoints)
    E = np.random.uniform(2, 8, size=sub_n_datapoints)

    C = np.random.uniform(25, 104, size=sub_n_datapoints)   
    D = 2*B   #Max 44, min 40
    Y1 = 3.5*A + 0.5*D 
    Y2 = - 2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([0, 0, 1, 0, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))

    return inputs, outputs


def intv_d_ood(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-3, 12, size=sub_n_datapoints)
    B = np.random.uniform(20, 22, size=sub_n_datapoints)
    E = np.random.uniform(2, 8, size=sub_n_datapoints)

    D = np.random.uniform(40, 44, size=sub_n_datapoints)   #Max 44, min 40
    C = 5*A + D   #Max 104, min. 25
    Y1 = 3.5*A + 0.5*D 
    Y2 = - 2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([0, 0, 0, 1, 0])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))

    return inputs, outputs


def intv_e_ood(sub_n_datapoints, intv_info = False):
    A = np.random.uniform(-3, 12, size=sub_n_datapoints)
    B = np.random.uniform(20, 22, size=sub_n_datapoints)
    E = np.random.uniform(2, 8, size=sub_n_datapoints)

    D = 2*B 
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = - 2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    if intv_info:
        intervention = np.array([0, 0, 0, 0, 1])
        intervention = np.tile(intervention, (sub_n_datapoints, 1))
        inputs = np.column_stack((intervention, inputs))

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

    # # Generate a random permutation of indices
    # indices = np.random.permutation(inputs.shape[0])

    # # Shuffle the rows of both arrays using the same indices
    # shuffled_inputs = inputs[indices]
    # shuffled_outputs = outputs[indices]

    return inputs, outputs
    



