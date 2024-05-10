import numpy as np 



def datapoint_gen(a, b, e):
    d = 2*b 
    c = 5*a + d
    y1 = 3.5*a + 0.5*d 
    y2 = -2*d + 0.2*e

    return c, d, y1, y2




def intv_a(sub_n_datapoints):

    A = np.random.uniform(-5, 10, size=sub_n_datapoints)
    B = np.random.uniform(5, 20, size=sub_n_datapoints)
    E = np.random.uniform(0, 15, size=sub_n_datapoints)

    D = 2*B
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = -2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs


def intv_b(sub_n_datapoints):
    A = np.random.uniform(-5, 10, size=sub_n_datapoints)
    B = np.random.uniform(5, 20, size=sub_n_datapoints)
    E = np.random.uniform(0, 15, size=sub_n_datapoints)

    D = 2*B
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = -2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs


def intv_c(sub_n_datapoints):
    A = np.random.uniform(-5, 10, size=sub_n_datapoints)
    B = np.random.uniform(5, 20, size=sub_n_datapoints)
    E = np.random.uniform(0, 15, size=sub_n_datapoints)

    C = np.random.uniform(-15, 90, size=sub_n_datapoints)
    D = 2*B
    Y1 = 3.5*A + 0.5*D 
    Y2 = -2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs


def intv_d(sub_n_datapoints):
    A = np.random.uniform(-5, 10, size=sub_n_datapoints)
    B = np.random.uniform(5, 20, size=sub_n_datapoints)
    E = np.random.uniform(0, 15, size=sub_n_datapoints)

    D = np.random.uniform(10, 40, size=sub_n_datapoints)
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = -2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs

def intv_e(sub_n_datapoints):
    A = np.random.uniform(-5, 10, size=sub_n_datapoints)
    B = np.random.uniform(5, 20, size=sub_n_datapoints)
    E = np.random.uniform(0, 15, size=sub_n_datapoints)

    D = 2*B
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = -2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs





def scm_intv_dataset_gen(n_datapoints, seed = 5):
    np.random.seed(seed)

    sub_n_datapoints = int(n_datapoints/5)

    I_A_in, I_A_out = intv_a(sub_n_datapoints)
    I_B_in, I_B_out = intv_b(sub_n_datapoints)
    I_C_in, I_C_out = intv_c(sub_n_datapoints)
    I_D_in, I_D_out = intv_d(sub_n_datapoints)
    I_E_in, I_E_out = intv_e(sub_n_datapoints)

    inputs = np.row_stack((I_A_in, I_B_in, I_C_in, I_D_in, I_E_in))
    outputs = np.row_stack((I_A_out, I_B_out, I_C_out, I_D_out, I_E_out))

    return inputs, outputs




def scm_intv_c_d_dataset_gen(n_datapoints, seed = 5):
    np.random.seed(seed)

    sub_n_datapoints = int(n_datapoints/2)

    I_C_in, I_C_out = intv_c(sub_n_datapoints)
    I_D_in, I_D_out = intv_d(sub_n_datapoints)

    inputs = np.row_stack((I_C_in, I_D_in))
    outputs = np.row_stack((I_C_out, I_D_out))

    return inputs, outputs






def intv_a_ood(n_out_of_domain):
    A = np.random.uniform(-3, 12, size=n_out_of_domain)  
    B = np.random.uniform(13, 25, size=n_out_of_domain)
    E = np.random.uniform(-2, 8, size=n_out_of_domain)  

    D = 2*B
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = -2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs


def intv_b_ood(n_out_of_domain):
    A = np.random.uniform(-3, 12, size=n_out_of_domain)  
    B = np.random.uniform(13, 25, size=n_out_of_domain)
    E = np.random.uniform(-2, 8, size=n_out_of_domain)  

    D = 2*B
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = -2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs


def intv_c_ood(n_out_of_domain):
    A = np.random.uniform(-3, 12, size=n_out_of_domain)  
    B = np.random.uniform(13, 25, size=n_out_of_domain)
    E = np.random.uniform(-2, 8, size=n_out_of_domain)   

    D = 2*B
    C = np.random.uniform(11, 110, size=n_out_of_domain)  
    Y1 = 3.5*A + 0.5*D 
    Y2 = -2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs


def intv_d_ood(n_out_of_domain):
    A = np.random.uniform(-3, 12, size=n_out_of_domain)  
    B = np.random.uniform(13, 25, size=n_out_of_domain)
    E = np.random.uniform(-2, 8, size=n_out_of_domain)   

    D = np.random.uniform(26, 50, size=n_out_of_domain)  
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = -2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs


def intv_e_ood(n_out_of_domain):
    A = np.random.uniform(-3, 12, size=n_out_of_domain)  
    B = np.random.uniform(13, 25, size=n_out_of_domain)
    E = np.random.uniform(-2, 8, size=n_out_of_domain)  

    D = 2*B
    C = 5*A + D
    Y1 = 3.5*A + 0.5*D 
    Y2 = -2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((Y1, Y2))

    return inputs, outputs





def scm_intv_ood(n_datapoints, seed = 5):
    np.random.seed(seed)

    sub_n_datapoints = int(n_datapoints/5)

    I_A_in, I_A_out = intv_a_ood(sub_n_datapoints)
    I_B_in, I_B_out = intv_b_ood(sub_n_datapoints)
    I_C_in, I_C_out = intv_c_ood(sub_n_datapoints)
    I_D_in, I_D_out = intv_d_ood(sub_n_datapoints)
    I_E_in, I_E_out = intv_e_ood(sub_n_datapoints)

    inputs = np.row_stack((I_A_in, I_B_in, I_C_in, I_D_in, I_E_in))
    outputs = np.row_stack((I_A_out, I_B_out, I_C_out, I_D_out, I_E_out))

    return inputs, outputs
    



