import numpy as np
import os
import shutil
import theta_init as theta_init
import random


class NeuralNetwork:
    def __init__(self, dim = None, norm_fcn = None, location = None):
        
        self.dim = dim
        if dim is None:
            print('failed, null dimension')
            quit()        
        self.leng = len(dim)

        if norm_fcn is None:
            norm_fcn = [sigmoid] * (self.leng - 1)
        self.norm_fcn = norm_fcn

        self.location = location
        if location is None or type(location) is not str:
            print('failed, null location')
            quit()

    def theta_generate(self, n = 1):

        """For initializing training set"""

        folder = self.location
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        for dataset in range(n):
            theta_init.create_file(self.dim,
                                   file_name = f"{self.location}/nn_theta_set_{dataset}.npz", ## Fix formating
                                   init_type = "normal")

    def theta_recover(self, i = 0):

        data = np.load(f'{self.location}/nn_theta_set_{i}.npz', allow_pickle=True)
        self.theta = [data[f'arr_{j}'] for j in range(self.leng - 1)]
        self.b = [data[f'arr_{j}'] for j in range(self.leng - 1, self.leng * 2 - 2)]

    def theta_save(self, i=0):
        np.savez(f'{self.location}/nn_theta_set_{i}.npz', *self.theta, *self.b)

    def theta_backup(self, i=0):
        data = np.load(f'{self.location}/nn_theta_set_{i}.npz', allow_pickle=True)
        theta = [data[f'arr_{j}'] for j in range(self.leng - 1)]
        b = [data[f'arr_{j}'] for j in range(self.leng - 1, self.leng * 2 - 2)]
        np.savez(f'{self.location}/nn_theta_set_{i+1}.npz', *theta, *b)


    def theta_single_use(self):
        
        self.theta = [theta_init.logistic_theta_init(0, 1, (self.dim[i], self.dim[i+1]))  for i in range(len(self.dim) - 1)]  
        self.b = [np.zeros((1,self.dim[i+1])) for i in range(len(self.dim) - 1)]


    def feedforward(self, X):
        '''Vaugely tested, probably good. Spits back out all the layers'''
        layers = []
        layers.append(X)
        for i in range(self.leng-1):
            X_hid = np.dot(layers[-1], self.theta[i]) + self.b[i]
            if type(self.norm_fcn[i]) == list:
                layers.append(np.column_stack([self.norm_fcn[i][j](X_hid[:, j]) for j in range(len(self.norm_fcn[i]))]))
            else:
                layers.append(self.norm_fcn[i](X_hid))

        return layers
    def feedforward_multi(self, X):
        '''Feedforward for multiple outputs. X is a list of inputs, each input is a 2D array. Returns a list of layers for each input.'''
        layers = [X]
        for i in range(self.leng - 1):
            X_hid = np.dot(layers[-1], self.theta[i]) + self.b[i]
            if type(self.norm_fcn[i]) == list:
                layers.append(np.column_stack(
                    [self.norm_fcn[i][j](X_hid[:, j]) for j in range(len(self.norm_fcn[i]))]
                ))
            else:
                layers.append(self.norm_fcn[i](X_hid))

        return layers

    def backward(self, X, y, learning_rate):

        layers = self.feedforward_multi(X)
        
        # Output layer error (delta) calculation
        if type(self.norm_fcn[-1]) == list:
            num_heads = len(self.norm_fcn[-1])
            delta_cols = []
            
            # Loop through the columns (Head 0 = Critic, Head 1 = Actor)
            for c in range(num_heads):
                # Grab the entire column (all 1000 frames) for this specific head
                y_col = y[:, c]
                layer_col = layers[-1][:, c]
                
                # Calculate the error and derivative for the whole batch at once
                error = y_col - layer_col
                derivative = self.norm_fcn[-1][c](layer_col, type='Derivative')
                
                # delta for this head is (error * derivative)
                delta_cols.append(error * derivative)
                
            # Stack the columns side-by-side to recreate the (Batch_Size, 2) matrix
            delta = [np.column_stack(delta_cols)]
            
        else:
            delta = [(y - layers[-1]) * self.norm_fcn[-1](layers[-1], type = 'Derivative')]

        for i in range(2, self.leng):
            delta.append(np.dot(delta[-1], self.theta[-i+1].T) * self.norm_fcn[-i](layers[-i], type = 'Derivative'))

        for i in range(self.leng-1):

            grad_theta = np.dot(layers[i].T, delta[-i-1])
            grad_b = np.sum(delta[-i-1], axis=0, keepdims=True)
            
            batch_size = len(X)
            grad_clip_limit = 5.0 * batch_size            
            grad_theta = np.clip(grad_theta, -grad_clip_limit, grad_clip_limit)
            grad_b = np.clip(grad_b, -grad_clip_limit, grad_clip_limit)
            
            self.theta[i] += grad_theta * learning_rate
            self.b[i] += grad_b * learning_rate

        return layers

    def take_it_back_now_yall(self, Y):
        '''anti feedward, if that makes sense'''
        layers = []
        layers.append(Y)
        for i in range(1, self.leng):
            X_hid = self.norm_fcn[-i](layers[0], type = 'inverse') - self.b[-i]
            layers.insert(0, np.dot(X_hid, self.theta[-i].T))
        return layers


    def train(self, X, y, epochs, learning_rate, cutoff_rate = 0, jumppy_learner = False, jumpy_index = (1, 1000)):
        loss = 100
        if jumppy_learner:
            learning_rate = 10**(-random.uniform(jumpy_index[0], jumpy_index[1]))

        for epoch in range(epochs):
            layers = self.backward(X, y, learning_rate) 
            if epoch % 2000 == 0:
                loss_new = np.mean(np.square(y - layers[-1]))
                if abs(loss - loss_new) <= cutoff_rate:
                    print(f'Epoch {epoch}, Loss:{loss_new}')
                    return
                loss = loss_new
                if jumppy_learner:
                    learning_rate = 10**(-random.uniform(jumpy_index[0], jumpy_index[1]))
                print(f'Epoch {epoch}, Loss:{loss}')
        loss_new = np.mean(np.square(y - layers[-1]))
        print(f'Epoch {epoch}, Loss:{loss_new}')


def sigmoid(x, type = 'Normal'):

    x_safe = np.clip(x, -500, 500)
    
    if type == 'Derivative':
        return x * (1 - x)
    if type == 'inverse':
        return np.log(x/(1-x))
    return 1 / (1 + np.exp(-x_safe))

def ReLU(x, type = 'Normal'):
    if type == 'Derivative':
        return 1 * (x > 0)
    return x * (x > 0)

def linear(x, type = 'Normal'):
    if type == 'Derivative':
        return np.ones(x.shape)
    return x




# NN = NeuralNetwork((6,5,5,4), [sigmoid, sigmoid, sigmoid], 'nn_library')
# NN.theta_generate()
# NN.theta_recover()
# print(NN.feedforward([np.array([[0,1], [1,0], [1,1]])])[-1])
# NN.train(np.array([[0,1], [1,0], [1,1]]), np.array([[0,1,0,1], [1,0,1,0], [1,1,0,0]]), 10000, 2)
# print(NN.feedforward([np.array([[0,1], [1,0], [1,1], [0,0]])])[-1])


# # Basic test:
# NN = NeuralNetwork((2,2,2), [ReLU, ReLU])
# NN.theta = [np.array([[1,1], [1,1]]), np.array([[1,1], [1,1]])]
# NN.b = [np.array([[0,0]]), np.array([[0,0]])]
# print(NN.feedforward([np.array([1.5, 0.5])]))

