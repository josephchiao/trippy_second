import numpy as np

def logistic_theta_init(loc, scale, size):

    return np.random.logistic(loc, scale, size)

def normal_theta_init(loc, scale, size):

    return np.random.normal(loc, scale, size)

def create_file(dim, file_name = "nn_theta_set.npz", init_type = "normal"):

    if init_type == "logistic":
        theta = [logistic_theta_init(0, 0.2, (dim[i], dim[i+1]))  for i in range(len(dim) - 1)]        

    if init_type == "normal":
        theta = [normal_theta_init(0, 0.2, (dim[i], dim[i+1]))  for i in range(len(dim) - 1)]        
    
    b = [np.zeros((1,dim[i])) for i in range(1, len(dim))]

    np.savez(file_name, *theta, *b)


