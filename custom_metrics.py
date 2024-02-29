import torch
import numpy as np


def CCC_loss(x, y):
    y = y.view(-1)
    x = x.view(-1)
    vx = x - torch.mean(x) 
    vy = y - torch.mean(y) 
    rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2)))+1e-8)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
    return 1 - ccc
    
    
def compute_VA_loss(Vout,Aout,label):
    bz,seq,_  = Vout.shape
    label = label.view(bz*seq,-1)
    Vout = Vout.view(bz*seq,-1)
    Aout = Aout.view(bz*seq,-1)
    ccc_loss = CCC_loss(Vout[:,0],label[:,0]) + CCC_loss(Aout[:,0],label[:,1])
    
    loss = ccc_loss 
    return loss, ccc_loss


def CCC_score(x, y):
    x = np.array(x)
    y = np.array(y)
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc