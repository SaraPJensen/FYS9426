import numpy as np 
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
from sklearn.preprocessing import MinMaxScaler



def datapoint_gen(a, b, e):
    d = 2*b 
    c = 5*a - a/d
    y1 = 3*a**2 + d**3 + a*d - d**2
    y2 = - d**2 + 4*d + np.sqrt(e)

    return c, d, y1, y2


def datapoint_gen_diff_rand_model(n_datapoints, a, b, e):

    c = np.random.uniform(-9.8, 19.6, size=n_datapoints)  #[-9.8, 19.6]
    d = np.random.uniform(6, 11, size=n_datapoints)   #[6, 11]

    y1 = 3*a**2 + d**3 + a*d - d**2
    y2 = - d**2 + 4*d + np.sqrt(e)

    return c, d, y1, y2


def datapoint_gen_diff_model(n_datapoints, a, b, e):

    d = 0.8*b + 0.5*a
    c = 3*a + 0.8*d
    #For now, C [-3, 14], D [0.6, 5.2], should be [-9.8, 19.6] and [6, 11]

    #Ensure that they are within the same range as the training data, to exclude this as a source of error
    C_scaler = MinMaxScaler(feature_range=(-9.77, 19.59))
    c = C_scaler.fit_transform(c.reshape(-1, 1)).reshape(-1)

    D_scaler = MinMaxScaler(feature_range=(6, 11))
    d = D_scaler.fit_transform(d.reshape(-1, 1)).reshape(-1)

    y1 = 3*a**2 + d**3 + a*d - d**2   #Now: [179.96, 1301.5]
    y2 = - d**2 + 4*d + np.sqrt(e)   #Now: [-75.84, -9.26]

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


def truncation(a_trunc, b_trunc, n_datapoints):
    loc = (a_trunc+b_trunc)/2
    sigma = abs(a_trunc-b_trunc)/5
    a, b = (a_trunc - loc)/sigma, (b_trunc - loc)/sigma
    dist = truncnorm.rvs(a, b, loc=loc, scale = sigma, size=n_datapoints) 

    # print(min(dist))
    # print(max(dist))
    # print(dist.shape)
    # print(np.mean(dist))
    # print()

    # ax.hist(dist, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    # plt.show()
    # plt.close()

    return dist


def scm_normal_dist(n_datapoints, seed = 5):
    np.random.seed(seed)
    inputs = np.zeros((n_datapoints, 5))
    outputs = np.zeros((n_datapoints, 2))

    A = truncation(-2, 4, n_datapoints)
    B = truncation(3, 5.5, n_datapoints)
    E = truncation(0, 8, n_datapoints)

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

    A = np.random.uniform(-3, 5, size=n_out_of_domain)
    B = np.random.uniform(0.5, 4.5, size=n_out_of_domain)
    E = np.random.uniform(2, 7, size=n_out_of_domain)

    # A = np.random.uniform(-4, 5, size=n_out_of_domain)
    # B = np.random.uniform(1, 3.4, size=n_out_of_domain)
    # E = np.random.uniform(3, 9, size=n_out_of_domain)  

    C, D, y1, y2 = datapoint_gen(A, B, E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((y1, y2))

    return inputs, outputs 


def scm_indep(n_points, seed = 5):
    np.random.seed(seed)
    inputs = np.zeros((n_points, 5))
    outputs = np.zeros((n_points, 2))

    A = np.random.uniform(-2, 4, size=n_points)
    B = np.random.uniform(3, 5.5, size=n_points)
    E = np.random.uniform(0, 8, size=n_points)

    C = np.random.uniform(-9.8, 19.6, size=n_points)  #[-9.7, 19.6]
    D = np.random.uniform(6, 11, size=n_points)   #[6, 11]
    
    y1 = 3*A**2 + D**3 + A*D - D**2
    y2 = - D**2 + 4*D + np.sqrt(E)  

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((y1, y2))

    return inputs, outputs 
      

def scm_indep_ood(n_ood, seed = 5):
    np.random.seed(seed)
    inputs = np.zeros((n_ood, 5))
    outputs = np.zeros((n_ood, 2))

    A = np.random.uniform(-4, 5, size=n_ood)
    B = np.random.uniform(1, 3.4, size=n_ood)
    E = np.random.uniform(3, 9, size=n_ood)  

    _, _, y1, y2 = datapoint_gen(A, B, E)

    C = np.random.uniform(-19.4, 24.3, size=n_ood)  
    D = np.random.uniform(2, 6.8, size=n_ood)   

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


     