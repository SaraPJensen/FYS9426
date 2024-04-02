import numpy as np



def normalize_data(inputs, targets):
    # Flatten the input and target arrays
    inputs = inputs.reshape(inputs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    # Calculate mean and standard deviation for normalization
    input_mean = np.mean(inputs, axis=0)
    input_std = np.std(inputs, axis=0)
    target_mean = np.mean(targets, axis=0)
    target_std = np.std(targets, axis=0)

    # Normalize the data
    inputs_normalized = (inputs - input_mean) / input_std
    targets_normalized = (targets - target_mean) / target_std

    return inputs_normalized, targets_normalized






def datapoint_gen(a, b, e):
    d = 2*b 
    c = 5*a + d
    y1 = 3.5*a + 0.5*d 
    y2 = - 2*d + 0.2*e

    return c, d, y1, y2




#Generate 500 data-points and store them in numpy arrays

#Input array: 500 x 5
#Output array: [y1, y2] -> 500 x 2 

n_datapoints = 10

def scm_dataset_gen(n_datapoints):
    np.random.seed(5)
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
print("Original outputs")
print(outputs)
print()

normalised_inputs, normalised_outputs = normalize_data(inputs, outputs)
print("Normalised outputs")
print(normalised_outputs)
print()

recalc_outputs = np.zeros((n_datapoints, 2))
for i in range(0, n_datapoints):
    a = normalised_inputs[i,0]
    b = normalised_inputs[i, 1]
    e = normalised_inputs[i, 4]
    c, d, y1, y2 = datapoint_gen(a, b, e)

    #inputs[i] = [A[i], B[i], c, d, E[i]]

    recalc_outputs[i] = [y1, y2]


print("Recalculated outputs")
print(recalc_outputs)
print()

inp, renormalised_outputs = normalize_data(inputs, recalc_outputs)
print("Renormalised outputs")
print(renormalised_outputs)

print("difference")
print(renormalised_outputs-normalised_outputs)


def scm_out_of_domain(n_out_of_domain):
        inputs = np.zeros((n_out_of_domain, 5))
        outputs = np.zeros((n_out_of_domain, 2))

        A = np.random.uniform(-3, 12, size=n_out_of_domain)
        B = np.random.uniform(20, 22, size=n_out_of_domain)
        E = np.random.uniform(2, 8, size=n_out_of_domain)  #Must be positive, since you cannot take the square root of a negative number!


        for i in range(0, n_out_of_domain):
            c, d, y1, y2 = datapoint_gen(A[i], B[i], E[i])

            inputs[i] = [A[i], B[i], c, d, E[i]]

            outputs[i] = [y1, y2]

        return inputs, outputs 


def scm_diff_seed(n_diff_seed):
        np.random.seed(5)
        inputs = np.zeros((n_diff_seed, 5))
        outputs = np.zeros((n_diff_seed, 2))

        A = np.random.uniform(-5, 10, size=n_diff_seed)
        B = np.random.uniform(5, 20, size=n_diff_seed)
        E = np.random.uniform(0, 15, size=n_diff_seed)

        for i in range(0, n_diff_seed):
            c, d, y1, y2 = datapoint_gen(A[i], B[i], E[i])

            inputs[i] = [A[i], B[i], c, d, E[i]]

            outputs[i] = [y1, y2]

        return inputs, outputs 




