import numpy as np

def logistic_theta_init(loc, scale, size):

    return np.random.logistic(loc, scale, size)

def normal_theta_init(loc, scale, size):

    return np.random.normal(loc, scale, size)

def create_file(dim, file_name = "nn_theta_set.npz", init_type = "normal", weight_loc = None, bias_loc = None, seed = None):

    if seed is not None and type(seed) is int:
        np.random.seed(seed)

    if weight_loc == None:
        weight_loc = np.zeros(len(dim) - 1)

    if init_type == "logistic":
        theta = [logistic_theta_init(weight_loc[i], 0.08, (dim[i], dim[i+1]))  for i in range(len(dim) - 1)]        

    if init_type == "normal":
        theta = [normal_theta_init(weight_loc[i], 0.08, (dim[i], dim[i+1]))  for i in range(len(dim) - 1)]        
    
    # b = [np.zeros((1,dim[i])) for i in range(1, len(dim))]
    if bias_loc == None:
        bias_loc = np.zeros(len(dim) - 1)
    b = [np.random.normal(bias_loc[i - 1], 0.05, (1,dim[i])) for i in range(1, len(dim))]

    np.savez(file_name, *theta, *b)


