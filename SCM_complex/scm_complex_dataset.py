import numpy as np 



def datapoint_gen(a, b, e):
    d = 2*b 
    c = 5*a - a/d
    y1 = 3*a**2 + d**3 + a*d - d**2
    y2 = - d**2 + 4*d + np.sqrt(e)

    return c, d, y1, y2


def datapoint_gen_diff_rand_model(n_datapoints, a, b, e):

    d = np.random.uniform(6, 11, size=n_datapoints)
    c = np.random.uniform(-9.8, 3.6, size=n_datapoints)

    y1 = 3*a**2 + d**3 + a*d - d**2
    y2 = - d**2 + 4*d + np.sqrt(e)

    return c, d, y1, y2


def datapoint_gen_diff_model(n_datapoints, a, b, e):

    d = 0.8*b - 0.5*a
    c = 3*a + 0.8*d
    y1 = 3*a**2 + d**3 + a*d - d**2
    y2 = - d**2 + 4*d + np.sqrt(e)

    return c, d, y1, y2



#Generate 500 data-points and store them in numpy arrays

#Input array: 500 x 5
#Output array: [y1, y2] -> 500 x 2 


def scm_dataset_gen(n_datapoints, seed = 5):
    np.random.seed(seed)
    inputs = np.zeros((n_datapoints, 5))
    outputs = np.zeros((n_datapoints, 2))

    A = np.random.uniform(-2, 4, size=n_datapoints)
    B = np.random.uniform(3, 5.5, size=n_datapoints)
    E = np.random.uniform(0, 8, size=n_datapoints)

    C, D, y1, y2 = datapoint_gen(A, B, E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((y1, y2))

    return inputs, outputs 




def scm_diff_seed(n_diff_seed, seed = 12345):
        np.random.seed(seed)
        inputs = np.zeros((n_diff_seed, 5))
        outputs = np.zeros((n_diff_seed, 2))

        A = np.random.uniform(-2, 4, size=n_diff_seed)
        B = np.random.uniform(3, 5.5, size=n_diff_seed)
        E = np.random.uniform(0, 8, size=n_diff_seed)

        C, D, y1, y2 = datapoint_gen(A, B, E)

        inputs = np.column_stack((A, B, C, D, E))
        outputs = np.column_stack((y1, y2))

        return inputs, outputs 




def scm_out_of_domain(n_out_of_domain, seed = 5):
        np.random.seed(seed)
        inputs = np.zeros((n_out_of_domain, 5))
        outputs = np.zeros((n_out_of_domain, 2))

        A = np.random.uniform(-4, 5, size=n_out_of_domain)
        B = np.random.uniform(1, 3.4, size=n_out_of_domain)
        E = np.random.uniform(3, 9, size=n_out_of_domain)  

        C, D, y1, y2 = datapoint_gen(A, B, E)

        inputs = np.column_stack((A, B, C, D, E))
        outputs = np.column_stack((y1, y2))

        return inputs, outputs 



def scm_diff_rand_model(n_diff_model, intv_info = False, seed = 5):
        np.random.seed(seed)
        inputs = np.zeros((n_diff_model, 5))
        outputs = np.zeros((n_diff_model, 2))

        A = np.random.uniform(-2, 4, size=n_diff_model)
        B = np.random.uniform(3, 5.5, size=n_diff_model)
        E = np.random.uniform(0, 8, size=n_diff_model) 

        C, D, y1, y2 = datapoint_gen_diff_rand_model(n_diff_model, A, B, E)

        inputs = np.column_stack((A, B, C, D, E))
        outputs = np.column_stack((y1, y2))

        if intv_info:
            intervention = np.array([0, 0, 0, 0, 0])
            intervention = np.tile(intervention, (n_diff_model, 1))
            inputs = np.column_stack((intervention, inputs))

        return inputs, outputs 



def scm_diff_model(n_diff_model, intv_info = False, seed = 5):
        np.random.seed(seed)
        inputs = np.zeros((n_diff_model, 5))
        outputs = np.zeros((n_diff_model, 2))

        A = np.random.uniform(-2, 4, size=n_diff_model)
        B = np.random.uniform(3, 5.5, size=n_diff_model)
        E = np.random.uniform(0, 8, size=n_diff_model)

        C, D, y1, y2 = datapoint_gen_diff_model(n_diff_model, A, B, E)

        inputs = np.column_stack((A, B, C, D, E))
        outputs = np.column_stack((y1, y2))

        if intv_info:
            intervention = np.array([0, 0, 0, 0, 0])
            intervention = np.tile(intervention, (n_diff_model, 1))
            inputs = np.column_stack((intervention, inputs))

        return inputs, outputs 




def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def Franke_data(n_dpoints, noise):
    np.random.seed(1234567)   #this was used for high noise

    x = np.arange(0,1,1/n_dpoints)
    y = np.arange(0,1,1/n_dpoints)

    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + noise*np.random.randn(n_dpoints, n_dpoints)

    x_flat = np.ravel(x)
    y_flat = np.ravel(y)
    z_flat = np.ravel(z)

    inputs = np.column_stack((x_flat, y_flat))
    print(inputs.shape)
    print(z_flat.shape)
    # exit()

    return inputs, z_flat



def super_simple(n_datapoints):
     
    A = np.random.uniform(-5, 10, size=n_datapoints)
    B = np.random.uniform(5, 20, size=n_datapoints)
    E = np.random.uniform(0, 15, size=n_datapoints)
 
    y1 = A
    y2 = B + E

    inputs = np.column_stack((A, B, E))
    outputs = np.column_stack((y1, y2))

    return inputs, outputs


     