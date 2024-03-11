import numpy as np 



def datapoint_gen(a, b, e):
    d = 2*b 
    c = 5*a + d/a
    y1 = 3*a**2 + d**3 + a*d - d**2
    y2 = - d**2 + 4*d + np.sqrt(e)

    #c = np.random.randint(0, 5)

    return c, d, y1, y2




#Generate 500 data-points and store them in numpy arrays

#Input array: 500 x 5
#Output array: [y1, y2] -> 500 x 2 

n_datapoints = 10

def scm_dataset_gen(n_datapoints):
    np.random.seed(2)
    inputs = np.zeros((n_datapoints, 5))
    outputs = np.zeros((n_datapoints, 2))

    A = np.random.uniform(-5, 10, size=n_datapoints)
    B = np.random.uniform(5, 20, size=n_datapoints)
    E = np.random.uniform(0, 15, size=n_datapoints)


    for i in range(0, n_datapoints):
        c, d, y1, y2 = datapoint_gen(A[i], B[i], E[i])

        inputs[i] = [A[i], B[i], c, d, E[i]]

        outputs[i] = [y1, y2]

    return inputs, outputs 



inputs, outputs = scm_dataset_gen(n_datapoints)



def smc_out_of_domain(n_out_of_domain):
        np.random.seed(5)
        inputs = np.zeros((n_out_of_domain, 5))
        outputs = np.zeros((n_out_of_domain, 2))

        # A = np.random.uniform(-3, 12, size=n_out_of_domain)
        # B = np.random.uniform(20, 22, size=n_out_of_domain)
        # E = np.random.uniform(2, 8, size=n_out_of_domain)  #Must be positive, since you cannot take the square root of a negative number!

        A = np.random.uniform(-5, 10, size=n_out_of_domain)
        B = np.random.uniform(5, 20, size=n_out_of_domain)
        E = np.random.uniform(0, 15, size=n_out_of_domain)


        for i in range(0, n_out_of_domain):
            c, d, y1, y2 = datapoint_gen(A[i], B[i], E[i])

            inputs[i] = [A[i], B[i], c, d, E[i]]

            outputs[i] = [y1, y2]

        return inputs, outputs 




