# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:58:17 2022

@author: Hylia
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def MHMC(var):
    burn = 100000
    sample = 50000
    freq = 10
    tau = var
    freqcounter = 0
    acceptance = 0
    scale = np.sqrt(tau)
    samples = np.array(())
    zn = 0
    for i in range(0, burn + sample):
        probz = sigdist(zn)
        znew = stats.norm.rvs(zn, scale, 1)[0]
        condznew = stats.norm.pdf(znew, zn, scale)
        condzold = stats.norm.pdf(zn, znew, scale) #technically the same value
        probznew = sigdist(znew)
        u = stats.uniform.rvs(0, 1, 1)[0]
        if (u <= np.exp(min(0, np.log(probznew) + np.log(condzold) - np.log(probz) - np.log(condznew)))):
            zn = znew
            acceptance = acceptance + 1
        if(i >= burn):
            if(freqcounter == freq):
                samples = np.append(samples, (zn))
                freqcounter = 0
            else:
                freqcounter = freqcounter + 1
    print(acceptance/(burn + sample))
    F = gass_hermite_quad(scalesig, 20, (10, 3))
    y = siggraph(F)
    x = np.linspace(-3, 3, 80)
    plt.hist(samples, bins = 50, density = True)
    plt.plot(x, y)
    plt.title("Tau = " + str(tau)) 
    plt.show()
    plt.close()
    return(acceptance/(burn+sample))

def Acctest():
    #swap the comments for plt.xlabel and y[j] to get the Metroplis-Hasting accuracy test
    y = np.zeros(5)
    #tau = (0.01, .1, .2, .5, 1) 
    eps = (0.005, 0.01, .1, .2, .5)
    #taulabels = ("0.01", ".1", ".2", ".5", "1")
    epslabels = (".005", ".01", ".1", ".2", ".5")
    for j in range(0, 5):
        #y[j] = MHMC(tau[j])
        y[j] = HMC(eps[j])
    #plt.bar(taulabels, y)
    plt.bar(epslabels, y)
    #plt.xlabel('tau')
    plt.xlabel('error')
    plt.show
        
    
def HMC(eps):        
    L = 10
    burn = 100000
    sample = 50000
    freq = 10
    samples = np.array(())
    acceptance = 0
    freqcounter = 0
    z0 = 0 #in this case z0 will be the new sample z at time t=0
    for i in range(0, burn + sample):
        # step 1
        r0 = np.random.normal(0, 1) #since it is not specified, the mass and s_i were set to 1
        #leapfrog step
        rt = r0
        zt = z0
        for j in range(0, L):
            rhalft = rt - eps/2 * (2 * zt - 10* (1-scalesig(zt, (10, 3)))) 
            zt = zt + eps * rhalft/1
            rt = rhalft - eps/2 * (2 * zt - 10* (1-scalesig(zt, (10, 3)))) 
        rt = -rt
        #acceptance prob
        u = np.random.uniform(0, 1)
        hamold = z0 ** 2 - np.log(scalesig(z0, (10, 3))) + r0 ** 2 /2 
        hamnew = zt ** 2 - np.log(scalesig(zt, (10, 3))) + rt ** 2 /2 
        if(u < np.exp(- hamnew + hamold)):
            z0 = zt
            acceptance = acceptance + 1
        if(i >= burn):
            if(freqcounter == freq):
                samples = np.append(samples, (zt))
                freqcounter = 0
            else:
                freqcounter = freqcounter + 1
    F = gass_hermite_quad(scalesig, 20, (10, 3))
    y = siggraph(F)
    x = np.linspace(-3, 3, 80)
    plt.hist(samples, bins = 50, density = True)
    plt.plot(x, y)
    plt.title("step size = " + str(eps)) 
    plt.show()
    plt.close()
    return(acceptance/(burn+sample))

def Normsamples(): #for problem 2.a
    mu = np.zeros(2)
    var = np.array(((3, 2.9), (2.9, 3)))
    samples = np.random.multivariate_normal(mu, var, size = 500)
    plt.scatter(samples[:,0], samples[:,1])

def Gibbs(): #f or problem 2.b
    mu = np.zeros(2)
    var = np.array(((3, 2.9), (2.9, 3)))
    samples = np.array([[-4., -4.]])
    z = np.array([-4., -4.])
    iterations = 100
    for i in range(0, iterations):
        z0mu = 0 + 2.9/3 * (z[1] - 0) #the conditional likelihood was already calculated in a previous homework
        z0var = (1- (2.9/3) ** 2) * 3 ** 2 #So I used wikipedia's conditional distribution for the sake of not redo-ing calculations
        z[0] = np.random.normal(z0mu, z0var)
        z1mu = 0 + 2.9/3 * (z[0] - 0) #since z0 and z1 have the same var, they have the same cond dist shape
        z1var = (1- (2.9/3) ** 2) * 3 ** 2
        z[1] = np.random.normal(z1mu, z1var)
        samples = np.append(samples, [z], axis = 0)
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.plot(samples[:,0], samples[:, 1])

def HMCmulti(U, gradU, z0, L, eps, burn, sampnum, freq):
    #to keep it simple, the distribution of r is Standard Normal Multivariate
    dim = len(z0)
    M = np.identity(dim) 
    rmu = np.zeros(dim)
    samples = np.empty((0, 4))
    rvecs = np.random.multivariate_normal(rmu, M, burn + sampnum)
    u = np.random.uniform(0, 1, burn + sampnum)
    zt = z0
    acceptance = 0
    for i in range(0, burn + sampnum):
        rt = rvecs[i]
        #Leapfrog step
        for j in range(0, L):
            rhalf = rt - (eps/2) * gradU(zt)
            zt = zt + eps * rhalf 
            rt = rhalf - (eps/2) * gradU(zt)
        Hold = U(z0) + .5 * rvecs[i] @ rvecs[i]
        Hnew = U(zt) + .5 * rt @ rt
        #Metroplis-Hasting step
        if(u[i] < min(1, np.exp(Hold - Hnew))):
            z0 = zt
            acceptance = acceptance + 1
        else:
            zt = z0 
        if(i >= burn and i % 10 == 0):
            samples = np.append(samples, [z0], axis = 0)
    finalsamples = samples 
    #return(finalsamples)
    return(finalsamples, acceptance/(burn + sampnum))

def Dist2(z):
    mean = (0, 0)
    Cov = ((3, 2.9), (2.9, 3)) 
    return(-stats.multivariate_normal.logpdf(z, mean, Cov))

def Dist2grad(z):
    Cov = np.array(((3, 2.9), (2.9, 3)))
    Covinv = np.linalg.inv(Cov)
    return(Covinv @ z)

#loading in training sets
with open("train.csv") as file_name:
    traindata = np.loadtxt(file_name, delimiter = ",")

with open("test.csv") as file_name:
    testdata = np.loadtxt(file_name, delimiter = ",")
trainphi = traindata[:,0:4]
testphi = testdata[:, 0:4]
traint = traindata[:, 4]
testt = testdata[:, 4]

def logDist3(z):
    #total = stats.multivariate_normal.logpdf(z, np.zeros(4), np.identity(4))
    total = -2 * math.log( 2 * math.pi) - z @ z /2
    y = sigmoid(trainphi @ z)
    total = total + (traint @ np.log(y) + (1 - traint) @ np.log(1-y))
    return(-total)
def logDist3grad(z):
    return(z + np.matmul((sigmoid(trainphi @ z) - traint).T, trainphi, dtype = 'float32')) #to squeeze out just a bit more speed
def sigmoid(x):
    return(1/(1 + np.exp(-x)))

def Banktest():
    eps = (.005, .01, .02, .05)
    L = (10, 20, 50)
    table = np.zeros((4, 5, 3))
    
    z0 = np.zeros(4)
    larget = 1 - np.repeat([testt], 1000, axis = 0).T #for use with predictive distribution
    for i in range(0, 3):
        for j in range(0, 4):
            samples, ac = HMCmulti(logDist3, logDist3grad, z0, L[i], eps[j], 10000, 10000, 10)
            sigsamp = sigmoid(testphi @ samples.T)
            #prediction accuracy
            predictions = np.where(sigsamp < .5, 0, 1) #gives the prediction for every entry
            predcomp = np.sum(predictions, axis = 1) - 1000 * testt #gives the number of incorrect predictions at every 6
            predacc = 1 - sum(abs(predcomp))/(1000 * 500) #averages tha amount of incorrect predictions and reverses it to get the accuracy
            #prediction likelihood
            likelihood = abs(sigsamp - larget)/1000 
            predlike= np.sum(np.sum(likelihood))/500
            table[i+1][j+1] = (ac, predacc, predlike)
            table[0][j+1] = eps[j]
            table[i+1][0] = L[i]
    print("accuracy")
    print(tabulate(table[:,:, 0], tablefmt = "latex"))
    print("prediction accuracy")
    print(tabulate(table[:,:, 1], tablefmt = "latex"))
    print("prediction likelihood")
    print(tabulate(table[:,:, 2], tablefmt = "latex"))

    


#Neural Network time
#I don't have much experience with NNs, so I based the construction of off PYtorch's tutorials
from torch.autograd import Variable
class Net(nn.Module):

    def __init__(self, layersize):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, layersize, bias = False) #the weights of the sigmoid function had no bias, so it makes sense to let the NN compete on even grounds
        self.fc2 = nn.Linear(layersize, layersize, bias = False)
        self.fc3 = nn.Linear(layersize, 1, bias = False)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.tanh(self.fc1(x)) #changing the activation type manually since it seems like more effort to automate it
        #x = F.tanh(self.fc2(x))
        x = self.fc3(x) #sigmoid will be done as part of the loss function
        return(x)
    
class BNet(nn.Module):
    def __init__(self, layer_size):
        super(BNet, self).__init__()
        self.layer_size = layer_size
        #layer 1 params
        self.weight_mu1 = nn.Parameter(torch.Tensor(layer_size, 4).normal_(0, 0.01))
        self.weight_rho1 = nn.Parameter(torch.Tensor(layer_size, 4).normal_(0, 0.01))
        #layer 2 params
        self.weight_mu2 = nn.Parameter(torch.Tensor(layer_size, layer_size).normal_(0, 0.01))
        self.weight_rho2 = nn.Parameter(torch.Tensor(layer_size, layer_size).normal_(0, 0.01))
        #layer 3 params
        self.weight_mu3 = nn.Parameter(torch.Tensor(1, layer_size).normal_(0, 0.01))
        self.weight_rho3 = nn.Parameter(torch.Tensor(1, layer_size).normal_(0, 0.01))
        
        #layer 1 bias
        self.bias_mu1 = nn.Parameter(torch.Tensor(layer_size).normal_(0, 0.01))
        self.bias_rho1 = nn.Parameter(torch.Tensor(layer_size).normal_(0, 0.01))
        #layer 2 bias
        self.bias_mu2 = nn.Parameter(torch.Tensor(layer_size).normal_(0, 0.01))
        self.bias_rho2 = nn.Parameter(torch.Tensor(layer_size).normal_(0, 0.01))
        #layer 3 bias
        self.bias_mu3 = nn.Parameter(torch.Tensor(1).normal_(0, 0.01))
        self.bias_rho3 = nn.Parameter(torch.Tensor(1).normal_(0, 0.01))
    def forward(self, x):
        eps1 = Variable(torch.normal(torch.zeros(self.layer_size, 4), torch.ones(self.layer_size, 4)))
        eps2 = Variable(torch.normal(torch.zeros(self.layer_size, self.layer_size), torch.ones(self.layer_size, self.layer_size)))
        eps3 = Variable(torch.normal(torch.zeros(1, self.layer_size), torch.ones(1, self.layer_size)))
        beps1 = Variable(torch.normal(torch.zeros(self.layer_size), torch.ones(self.layer_size)))
        beps2 = Variable(torch.normal(torch.zeros(self.layer_size), torch.ones(self.layer_size)))
        beps3 = Variable(torch.normal(torch.zeros(1), torch.ones(1)))
        self.weight1 = self.weight_mu1 + eps1 * torch.sqrt(torch.log(1 + torch.exp(self.weight_rho1)))
        self.weight2 = self.weight_mu2 + eps2 * torch.sqrt(torch.log(1 + torch.exp(self.weight_rho2)))
        self.weight3 = self.weight_mu3 + eps3 * torch.sqrt(torch.log(1 + torch.exp(self.weight_rho3)))
        self.bias1 = self.bias_mu1 + beps1 * torch.sqrt(torch.log(1 + torch.exp(self.bias_rho1)))
        self.bias2 = self.bias_mu2 + beps2 * torch.sqrt(torch.log(1 + torch.exp(self.bias_rho2)))
        self.bias3 = self.bias_mu3 + beps3 * torch.sqrt(torch.log(1 + torch.exp(self.bias_rho3)))
        x = F.relu(torch.matmul(x, self.weight1.T) + self.bias1)
        x = F.relu(torch.matmul(x, self.weight2.T) + self.bias2)
        #x = torch.tanh(torch.matmul(x, self.weight1.T))
        #x = torch.tanh(torch.matmul(x, self.weight2.T))
        x = torch.matmul(x, self.weight3.T + self.bias3) 
        x = torch.flatten(x)
        #x = torch.sigmoid(x)
        return(x)
        
    def addon(self):
        lpw1 = -.5 * np.log(2 * np.pi)  - 1/2 * (torch.log(1+self.weight_rho1) ** 2 + self.weight_mu1 ** 2) #prior log exo value w/ standard normal gauss prior
        lpw2 = -.5 * np.log(2 * np.pi)  - 1/2 * (torch.log(1+self.weight_rho2) ** 2 + self.weight_mu2 ** 2)
        lpw3 = -.5 * np.log(2 * np.pi)  - 1/2 * (torch.log(1+self.weight_rho3) ** 2 + self.weight_mu3 ** 2) 
        #the constants don't technically matter here, since we really only need the gradient of this function
        
        blpw1 = -.5 * np.log(2 * np.pi)  - 1/2 * (torch.log(1+self.bias_rho1) ** 2 + self.bias_mu1 ** 2)
        blpw2 = -.5 * np.log(2 * np.pi)  - 1/2 * (torch.log(1+self.bias_rho2) ** 2 + self.bias_mu2 ** 2)
        blpw3 = -.5 * np.log(2 * np.pi) - 1/2 * (torch.log(1+self.bias_rho3) ** 2 + self.bias_mu3 ** 2)
        
        lqw1 = np.log(2 * np.pi * np.e)  + torch.log(torch.log(1 + torch.exp(self.weight_rho1))) #entropy for posterior
        lqw2 = np.log(2 * np.pi * np.e)  + torch.log(torch.log(1 + torch.exp(self.weight_rho2)))
        lqw3 = np.log(2 * np.pi * np.e)  + torch.log(torch.log(1 + torch.exp(self.weight_rho3)))
        
        blqw1 = np.log(2 * np.pi * np.e) + torch.log(torch.log(1 + torch.exp(self.bias_rho1)))
        blqw2 = np.log(2 * np.pi * np.e) + torch.log(torch.log(1 + torch.exp(self.bias_rho2)))
        blqw3 = np.log(2 * np.pi * np.e) + torch.log(torch.log(1 + torch.exp(self.bias_rho3)))
        
        lpw = torch.sum(lpw1) + torch.sum(lpw2) + torch.sum(lpw3) + torch.sum(blpw1) + torch.sum(blpw2) + torch.sum(blpw3)
        lqw = torch.sum(lqw1) + torch.sum(lqw2) + torch.sum(lqw3) + torch.sum(blqw1) + torch.sum(blqw2) + torch.sum(blqw3)
        
        return(lpw, lqw)
        
def criterion(out, y, net):
    lpw, lqw = net.addon()
    eps = 10 ** -8 #for stability
    loglike = y * torch.log(torch.sigmoid(out) + eps) + (1 - y) * torch.log(1 - torch.sigmoid(out) + eps)
    loss = (-lpw + -lqw - torch.sum(loglike))
    return(loss)
        
        
        
    
'''
floatphi = trainphi.astype('float32')
tensorphi = torch.from_numpy(floatphi)
tensorphi = Variable(torch.reshape(tensorphi, (872, 1, 4))) #formatting the input to be the right format
floatt = traint.astype('float32')
tensort = Variable(torch.from_numpy(floatt))
bnet = trainBNN(10, tensorphi, tensort, 10 ** -5, 1000)
'''
def trainBNN(nodes, phi, y, learn, iters):
    #params are automatically generated as random normal, so I don't need to generate them
    net = BNet(nodes)
    optimizer = optim.Adam(net.parameters(), lr = learn) 
    for i in range(0, iters): #skip stochastic as noted in the question
        optimizer.zero_grad()
        out = net(phi)
        loss = criterion(out, y, net)
        print(loss)
        loss.backward()        
        optimizer.step()
    return(net)
        
'''
out = bnet(tensorphi)
pred = torch.zeros(out.size())
for i in range(0, len(out)):
    if(out[i] > .5):
        pred[i] = 1
    else:
        pred[i] = 0
sum(abs(pred - tensort))
'''
    
    
    
    
'''code to get the graph of HMC samples for problem 2.b
z0 = np.array([-4, -4])
samples = HMCmulti(Dist2, Dist2grad, z0, 20, .1, 0, 100, 1)
plt.plot(samples[:,0], samples[:,1])
plt.scatter(samples[:,0], samples[:,1])
'''            

'''
code for problem 3
z0 = np.zeros(4)
samples2= HMCmulti(logDist3, logDist3grad, z0, 20, .01, 100000, 10000, 10)
        
'''            
    

    
def scalesig(x, params):
    a = params[0]
    b = params[1]
    return 1 / (1 + np.e ** (-1 * (a * x + b)))

def sigdist(z):
    y = np.exp(- z ** 2) * scalesig(z, (10, 3)) 
    return(y)




#Using from last homework to generate the original plot
def gass_hermite_quad( f, degree, params):
    '''
    Calculate the integral (1) numerically.
    :param f: target function, takes a array as input x = [x0, x1,...,xn], and return a array of function values f(x) = [f(x0),f(x1), ..., f(xn)]
    :param degree: integer, >=1, number of points
    :return:
    '''

    points, weights = np.polynomial.hermite.hermgauss( degree)

    #function values at given points
    f_x = f(points, params)

    #weighted sum of function values
    F = np.sum( f_x  * weights)

    return F
'''
if __name__ == '__main__':

    # Example 1, f(x) = x^2, degree = 3, whose closed form solution is sqrt(pi) / 2
    def x_square( x):
        return x* x
    F = gass_hermite_quad( x_square,3 )
    print( F)

    # Example 2, f(x) = x * sin x, degree = 10, whose closed form solution is sqrt( pi) / e^(1/4) / 2)
    def my_func(x):
        return x * np.sin( x)

    F = gass_hermite_quad( my_func, degree=10)
    print(F)
'''   
def scalesig(x, params):
    a = params[0]
    b = params[1]
    return 1 / (1 + math.e ** (-1 * (a * x + b)))

def siggraph(F):
    x = np.linspace(-5, 5, 80)
    y = np.exp(- x ** 2) * scalesig(x, (10, 3)) / F
    return(y)