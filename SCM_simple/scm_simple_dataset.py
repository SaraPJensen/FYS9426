import numpy as np 
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
from sklearn.preprocessing import MinMaxScaler



def datapoint_gen(a, b, e):
    d = 2*b 
    c = 5*a + d
    y1 = 3.5*a + 0.5*d 
    y2 = -2*d + 0.2*e

    return c, d, y1, y2


def datapoint_gen_diff_rand_model(n_datapoints, a, b, e):

    c = np.random.uniform(-15, 90, size=n_datapoints)  
    d = np.random.uniform(10, 40, size=n_datapoints) 

    y1 = 3.5*a + 0.5*d 
    y2 = -2*d + 0.2*e

    return c, d, y1, y2


def datapoint_gen_diff_model(n_datapoints, a, b, e):

    d = 0.8*b - 0.5*a
    c = 3*a - 0.8*d

    #Ensure that they are within the same range as the training data, to exclude this as a source of error
    C_scaler = MinMaxScaler(feature_range=(-15, 90))   
    c = C_scaler.fit_transform(c.reshape(-1, 1)).reshape(-1)

    D_scaler = MinMaxScaler(feature_range=(10, 40))
    d = D_scaler.fit_transform(d.reshape(-1, 1)).reshape(-1)

    y1 = 3.5*a + 0.5*d 
    y2 = -2*d + 0.2*e

    # y1 = 3*a**2 + d**3 + a*d - d**2   #Now: [179.96, 1301.5]  XXX
    # y2 = - d**2 + 4*d + np.sqrt(e)   #Now: [-75.84, -9.26] XXX

    return c, d, y1, y2



def scm_dataset_gen(n_datapoints, seed = 5):
    np.random.seed(seed)
    inputs = np.zeros((n_datapoints, 5))
    outputs = np.zeros((n_datapoints, 2))

    A = np.random.uniform(-5, 10, size=n_datapoints)
    B = np.random.uniform(5, 20, size=n_datapoints)
    E = np.random.uniform(0, 15, size=n_datapoints)

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


def scm_normal_dist(n_datapoints, seed = 5):  #Might not be necessary here
    np.random.seed(seed)
    inputs = np.zeros((n_datapoints, 5))
    outputs = np.zeros((n_datapoints, 2))

    A = truncation(-5, 10, n_datapoints)
    B = truncation(5, 20, n_datapoints)
    E = truncation(0, 15, n_datapoints)

    C, D, y1, y2 = datapoint_gen(A, B, E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((y1, y2))

    return inputs, outputs 




def scm_diff_seed(n_diff_seed, seed = 12345):
    np.random.seed(seed)
    inputs = np.zeros((n_diff_seed, 5))
    outputs = np.zeros((n_diff_seed, 2))

    A = np.random.uniform(-5, 10, size=n_diff_seed)
    B = np.random.uniform(5, 20, size=n_diff_seed)
    E = np.random.uniform(0, 15, size=n_diff_seed)

    C, D, y1, y2 = datapoint_gen(A, B, E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((y1, y2))

    return inputs, outputs 




def scm_out_of_domain(n_out_of_domain, seed = 5):
    np.random.seed(seed)
    inputs = np.zeros((n_out_of_domain, 5))
    outputs = np.zeros((n_out_of_domain, 2))

    A = np.random.uniform(-3, 12, size=n_out_of_domain)  
    B = np.random.uniform(13, 25, size=n_out_of_domain)
    E = np.random.uniform(-2, 8, size=n_out_of_domain)

    C, D, y1, y2 = datapoint_gen(A, B, E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((y1, y2))

    return inputs, outputs 


def scm_indep(n_points, seed = 5):
    np.random.seed(seed)
    inputs = np.zeros((n_points, 5))
    outputs = np.zeros((n_points, 2))

    A = np.random.uniform(-5, 10, size=n_points)
    B = np.random.uniform(5, 20, size=n_points)
    E = np.random.uniform(0, 15, size=n_points)

    C = np.random.uniform(-15, 90, size=n_points)
    D = np.random.uniform(10, 40, size=n_points)
    
    y1 = 3.5*A + 0.5*D
    y2 = - 2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((y1, y2))

    return inputs, outputs 
      


def scm_indep_ood(n_ood, seed = 5):
    np.random.seed(seed)
    inputs = np.zeros((n_ood, 5))
    outputs = np.zeros((n_ood, 2))

    A = np.random.uniform(-3, 12, size=n_ood)  
    B = np.random.uniform(9, 22, size=n_ood)
    E = np.random.uniform(2, 8, size=n_ood)  

    C = np.random.uniform(19, 104, size=n_ood)  
    D = np.random.uniform(18, 44, size=n_ood)   

    y1 = 3.5*A + 0.5*D
    y2 = - 2*D + 0.2*E

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((y1, y2))

    return inputs, outputs 
     



def scm_diff_rand_model(n_diff_model, seed = 5):
    np.random.seed(seed)
    inputs = np.zeros((n_diff_model, 5))
    outputs = np.zeros((n_diff_model, 2))

    A = np.random.uniform(-5, 10, size=n_diff_model)
    B = np.random.uniform(5, 20, size=n_diff_model)
    E = np.random.uniform(0, 15, size=n_diff_model)

    C, D, y1, y2 = datapoint_gen_diff_rand_model(n_diff_model, A, B, E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((y1, y2))

    return inputs, outputs 



def scm_diff_model(n_diff_model, seed = 5):
    np.random.seed(seed)
    inputs = np.zeros((n_diff_model, 5))
    outputs = np.zeros((n_diff_model, 2))

    A = np.random.uniform(-5, 10, size=n_diff_model)
    B = np.random.uniform(5, 20, size=n_diff_model)
    E = np.random.uniform(0, 15, size=n_diff_model)

    C, D, y1, y2 = datapoint_gen_diff_model(n_diff_model, A, B, E)

    inputs = np.column_stack((A, B, C, D, E))
    outputs = np.column_stack((y1, y2))

    return inputs, outputs 



