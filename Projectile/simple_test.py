import numpy as np 



g = 9.81

def Position(x_0, y_0, v_x0, v_y0, t):
    x = x_0 + v_x0 * t
    y = y_0 + v_y0*t - 0.5*g*t**2 

    range = v_x0*v_y0/g
    max_h = v_y0**2/(2*g)
    max_t = 2*v_y0/g

    if y < 0:
        y = 0
        x = v_x0*v_y0/g   

    return x, y


def Extremals(x_0, y_0, v_x0, v_y0):
    range = v_x0*v_y0/g   
    max_h = v_y0**2/(2*g)   
    max_t = 2*v_y0/g

    return range, max_h, max_t


""" x_0 = 0
y_0 = 15
v_x0 = 15
v_y0 = 20
t = 5

print("")

position = tuple(round(element, 3) for element in Position(x_0, y_0, v_x0, v_y0, t))

print("Position after 5 sec: ", position)

extremes = tuple(round(element, 3) for element in Extremals(x_0, y_0, v_x0, v_y0, t))

print("")
print("Extremals", 
      "\nRange: ", extremes[0], 
      "\nMax height: ", extremes[1], 
      "\nMax time: ", extremes[2]) """


"""
Domain for the different input variables

x_0: [0, 10]
y_0: [0, 15]

v_x0: [0, 15]
v_y0: [0, 20]

t: [0, 5]

"""

#Generate 500 data-points and store them in numpy arrays
#Save the position every second 
#Input array: 500 x 4
#Output array: [range, max_h, max_t, 10 time steps] -> 500 x 13 

n_datapoints = 10

def Dataset(n_datapoints, max_time,  n_timesteps):

    inputs = np.zeros((n_datapoints, 4))
    extremals = np.zeros((n_datapoints, 3))
    trajectories = np.zeros((n_datapoints, n_timesteps, 2))

    x_0s = np.random.uniform(0.0, 10.0, size = n_datapoints)
    y_0s = np.random.uniform(0.0, 15.0, size = n_datapoints)
    v_x0s = np.random.uniform(0.0, 15.0, size = n_datapoints)
    v_y0s = np.random.uniform(0.0, 20.0, size = n_datapoints)
    ts = np.linspace(0.0, max_time, n_timesteps)  #Forget about the precise trajectories for now


    for i in range(0, n_datapoints):
        inputs[i] = [x_0s[i], y_0s[i], v_x0s[i], v_y0s[i]]

        extremals[i] = Extremals(x_0s[i], y_0s[i], v_x0s[i], v_y0s[i])

        for t_idx, t in enumerate(ts): 
            trajectories[i, t_idx, :] = Position(x_0s[i], y_0s[i], v_x0s[i], v_y0s[i], t)


    return inputs, extremals, trajectories


""" inputs, extremals, trajectories = Dataset(10, 5, 10)
            
print(inputs)
print()
print(extremals)
print()
print(trajectories) """




