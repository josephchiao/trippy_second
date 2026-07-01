import numpy as np
from matplotlib import pyplot as plt
import math 

class pid_controller:
    def __init__(self, target, location, kp, ki, kd, display = False, mode = 'normal'):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.location = location
        self.prev_error = 0
        self.integral = 0
        self.display = display
        self.mode = mode

    def update(self):
        
        if self.mode == 'normal':
            output = self.proportional() + self.integrate() - self.derivative()
        elif self.mode == "curved":
            output = self.proportional()**3 + self.integrate() - self.derivative()
        if self.display == True:
            print('p', self.proportional())
            print('i', self.integrate())
            print('d', self.derivative())
        if self.prev_error * (self.target - self.location) < 0:
            self.integral = 0
            self.prev_error = self.target - self.location
        else:
            self.prev_error = self.target - self.location
            self.integral += self.prev_error
        return output

    def proportional(self):
        return self.kp * (self.target - self.location)
    
    def integrate(self):
        return self.ki * self.integral
    
    def derivative(self):
        return self.kd * (self.prev_error - (self.target - self.location))
    
