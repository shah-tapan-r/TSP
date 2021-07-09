import torch
import pickle
import pandas as pd
import glob
import shutil
# import psutil
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse
# import torchsort
from torch.utils.data import Dataset, TensorDataset
from scipy.optimize import minimize,optimize
from torch.utils.data import DataLoader
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# poly = PolynomialFeatures(degree=2)
# X_ = poly.fit_transform(X)
# model = LinearRegression()
# model.fit(X_,Y)
# Ypred = model.predict(X_)
# plt.plot(Y,Ypred)

from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import operator as op
from functools import reduce
import warnings
warnings.filterwarnings('ignore')


def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def perm2mat(x):
    pm = []
    zeros = np.zeros(len(x))
    for xx in x:
        np.put(zeros, [xx], 1)
        pm.append(zeros)
        zeros = np.zeros(len(x))
    return np.vstack(pm)

def mat2perm(x):
    x = x.argsort(axis=1)
    perm = []
    for xind in range(x.shape[0]):
        init=0
        while(x[xind,init]  in perm):
            init+=1
        perm.append(x[xind,init])
    return perm

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

def train(model, train_loader, optimizer, epoch, quiet=False, grad_clip=None):
    model.train()

    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))
    losses = OrderedDict()
    for x,y in train_loader:
        if(x.shape[0]>1):
            x = x.cuda()
            y=y.cuda()
            out = model.loss(x,y)
            optimizer.zero_grad()
            out['loss'].backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            for k, v in out.items():
                if k not in losses:
                    losses[k] = []
                losses[k].append(v.item())
                avg_loss = np.mean(losses[k][-50:])

    if not quiet:
        pbar.close()
    return losses


def eval_loss(model, data_loader, quiet):
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        z_comp = []
        y_pred_comp = []
        for x,y in data_loader:
            x = x.cuda()
            y = y.cuda()
            out = model.loss(x,y)
            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = 'Test '
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f', {k} {total_losses[k]:.4f}'
        if not quiet:
            print(desc)
    return total_losses


def train_epochs(model, train_loader, test_loader=None, train_args=None, quiet=False):
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = OrderedDict(), OrderedDict()

    best_test_loss = 100000000
    for epoch in range(epochs):
        model.train()
        train_loss = train(model, train_loader, optimizer, epoch, quiet, grad_clip)
        if(test_loader is not None):
            if(epoch==(epochs-1)):
                test_loss = eval_loss(model, test_loader, quiet)
            else:
                test_loss = eval_loss(model, test_loader, quiet)


        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
                test_losses[k] = []
            train_losses[k].extend(train_loss[k])
            if (test_loader is not None):
                test_losses[k].append(test_loss[k])
        if (test_loader is not None):
            if(np.min(test_losses['loss'])<best_test_loss):
                best_test_loss = np.min(test_losses['loss'])
                best_weights = model.state_dict()

    if (test_loader is not None):
        model.load_state_dict(best_weights)

        return model,train_losses, test_losses
    else:
        model.eval()
        return model


def poly(x,d):
    poly = PolynomialFeatures(degree=d,include_bias=False)
    X_ = poly.fit_transform(x.detach().cpu().numpy())
    return torch.from_numpy(X_)

def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()

def weighted_mse_loss(input, target):
    weight = torch.abs(target-target.max())
    return (weight * (input - target)**2).mean()

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        self.base_size = (128, int(np.floor(output_shape[1] / 8)+1) , int(np.floor(output_shape[2] / 8)+1) )
        self.fc = nn.Linear(latent_dim, int(np.prod(self.base_size)))
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_shape[0], 3, padding=1),
            nn.Upsample(size=(output_shape[1],output_shape[2]))
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], *self.base_size)
        out = self.deconvs(out)
        return out


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.convs = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
        )
        conv_out_dim = (np.floor(input_shape[1] / 8)+1) * (np.floor(input_shape[1] / 8)+1) * 256
        self.fc = nn.Linear(int(conv_out_dim), 2 * latent_dim)

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.shape[0], -1)
        mu, log_std = self.fc(out).chunk(2, dim=1)
        return mu, log_std


class ConvVAE(nn.Module):
    def __init__(self, input_shape, latent_size,degree=2):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.encoder = ConvEncoder(input_shape, latent_size)
        self.decoder = ConvDecoder(latent_size, input_shape)
        self.lr = nn.Linear(ncr(latent_size + degree, degree) - 1, 1, bias=True)
        self.poly = poly
        self.degree = degree
        self.relu = nn.ReLU()

    def predict(self, z):
        ##Make polynomial features
        poly_z = self.poly(z, self.degree).cuda()
        yout = self.relu(self.lr((poly_z)))
        return yout

    def loss(self, x,y):
        x = 2 * x - 1
        x_noise = x+(0.1**0.5)*torch.randn(x.shape).to(device)
        mu, log_std = self.encoder(x_noise)
        z = torch.randn_like(mu) * log_std.exp() + mu
        z_noise = z+mu
        yout = self.predict(z)
        x_recon = self.decoder(z_noise)

        recon_loss = F.mse_loss(x, x_recon, reduction='none').view(x.shape[0], -1).sum(1).mean()
        kl_loss = -log_std - 0.5 + (torch.exp(2 * log_std) + mu ** 2) * 0.5
        kl_loss = kl_loss.sum(1).mean()

        if(np.random.random()>0.5):
            reg_loss =  nn.MSELoss(size_average='mean')(y,yout.squeeze())
        else:
            reg_loss = nn.L1Loss(size_average='mean')(y, yout.squeeze())



        return OrderedDict(loss=recon_loss + kl_loss+reg_loss, recon_loss=recon_loss,
                           kl_loss=kl_loss,reg_loss = reg_loss)

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.latent_size).cuda()
            samples = torch.clamp(self.decoder(z), -1, 1)
        return samples.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5


def main():


    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--launch",type=str,default="1", help="" )

    args = parser.parse_args()
    PROJECT = 'EXECUTION-'+args.launch
    MODELPATH = os.path.join(PROJECT,'models')
    if(not os.path.exists(MODELPATH)):
        os.mkdir(MODELPATH)
    INPUTPATH = os.path.join(PROJECT,'SURROGATE-OUT')
    OUTPATH = os.path.join(PROJECT,'SURROGATE-IN')
    inputfiles = glob.glob(INPUTPATH+'/*.txt')
    outfile = OUTPATH+'/seed.txt'
    modelfile = MODELPATH+'/model.pkl'
    li = []
    try:
        for filename in inputfiles:
            df = pd.read_csv(filename,delim_whitespace=False,header=None)
            li.append(df)
            if(df.shape[0]>100000):
                os.remove(filename)


        dt = pd.concat(li,axis=0,ignore_index=True)


        if(dt.shape[0]>1000):
            print('Starting training the convex surrogate model with {} samples'.format(dt.shape[0]))


            BATCH_SIZE = 128
            N_EPOCHS = 20
            # dt = dt.iloc[:10000, :]
            ycol = dt.columns[-1]
            xcols = dt.columns[1:-1]
            input_dim = len(xcols)
            X = dt[xcols]
            Y = dt[ycol]
            Y = (np.max(Y) - Y) / (np.max(Y) - np.min(Y))
            X = np.apply_along_axis(perm2mat, 1, X.values)
            X = X[:, np.newaxis, :, :]
            np.random.seed(10102)
            train_index = np.random.choice(range(X.shape[0]), int(0.8 * X.shape[0]))
            test_index = list(set(range(len(Y))) - set(train_index))

            train_dataset = TensorDataset(torch.from_numpy(X[train_index, :]).float(),
                                          torch.from_numpy(Y.values[train_index]).float())
            test_dataset = TensorDataset(torch.from_numpy(X[test_index, :]).float(),
                                         torch.from_numpy(Y.values[test_index]).float())
            train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            if(os.path.exists(modelfile)):
                model = pickle.load(open(modelfile,'rb'))
            else:
                model = ConvVAE((1, input_dim, input_dim), 16).cuda()
            model, train_losses, test_losses = train_epochs(model, train_iterator, test_iterator,
                                                            dict(epochs=N_EPOCHS, lr=0.0001, grad_clip=1), quiet=False)
            pickle.dump(model,open(modelfile,'wb'))
            best_loss_ind = np.argmin(test_losses['loss'])

            # results[file]['reg_loss'].append(test_losses['reg_loss'][best_loss_ind])
            # results[file]['recon_loss'].append(test_losses['recon_loss'][best_loss_ind])
            # results[file]['kl_loss'].append(test_losses['kl_loss'][best_loss_ind])
            # results[file]['n_training_points'].append(ind)
            # pickle.dump(results,open('results.pkl','wb'))

            def f_conv(x):
                x = np.array(x).reshape(1, -1)
                return model.predict(torch.from_numpy(x).float()).detach().cpu().numpy()[0, 0]

            # # x0=model.encode(torch.from_numpy(np.random.random(20).reshape(1,-1)).float().cuda()).detach().cpu().numpy()
            z0 = model.encoder(torch.from_numpy(2 * X[-1, :].reshape(1, 1, input_dim, -1) - 1).float().cuda())[
                0].detach().cpu().numpy()
            # # res = optimize.brute(f_conv,ranges=((-4,4),(-4,4)))
            res = optimize.fmin_powell(f_conv, x0=z0)
            z = torch.from_numpy(res.reshape(1, -1)).float().cuda()
            x_r = model.decoder(z).detach().cpu().numpy().reshape(-1, input_dim)
            perm = mat2perm(x_r)
            if (os.path.exists(outfile)):
                seeds = pd.read_csv(outfile, header=None,index_col=0)
                seeds = seeds.rename(columns=dict(zip(seeds.columns,seeds.columns-1)))
                seeds = seeds.append(pd.DataFrame({seeds.index[-1] + 1:  perm}).T,ignore_index=True)

            else:
                seeds = pd.DataFrame({0: perm}).T
            seeds.to_csv(outfile, header=None,index=True)
            return 0
        else:
            return 2
    except Exception as e:
        print(e)
        return 1



if __name__=='__main__':

    ps = 0
    while(True):
        old_ps = ps
        ps= main()









    #
    # def f(X):
    #     return ((X[0] * X[1]) + 0.2538) ** 2 + ((X[1] * X[2]) - 0.3196) ** 2 + ((X[2] * X[3]) - 0.2448) ** 2 + ((X[3] * X[4]) + 0.306) ** 2 + (
    #                 (X[4] * X[5]) - 0.1445) ** 2 \
    #            + ((X[5] * X[6]) - 0.0323) ** 2 + ((X[6] * X[7]) + 0.1748) ** 2 + ((X[7] * X[8]) + 0.6164) ** 2 + ((X[8] * X[9]) - 0.1876) ** 2 + (
    #                        (X[9] * X[10]) -
    #                        0.2716) ** 2 + ((X[10] * X[11]) - 0.8536) ** 2 + ((X[11] * X[12]) + 0.66) ** 2 + ((X[12] * X[13]) + 0.0375) ** 2 + ((X[13] * X[14]) -
    #                                                                                                                                            0.0435) ** 2 + (
    #                        (X[14] * X[15]) + 0.8004) ** 2 + ((X[15] * X[16]) - 0.23) ** 2 + ((X[16] * X[17]) + 0.2175) ** 2 + (
    #                        (X[17] * X[18]) - 0.1218) ** 2 + ((X[18] *
    #                                                           X[19]) - 0.1092) ** 2 + ((X[19] * X[0]) - 0.4212) ** 2
    # f(x_r)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # z1 = z2 = np.arange(-4.0, 4.0, 0.05)
    # Z1, Z2 = np.meshgrid(z1, z2)
    # Z =np.concatenate([np.ravel(Z1).reshape(-1, 1),np.ravel(Z2).reshape(-1, 1)],
    #                                                    axis=1)
    # ys = model.predict(torch.from_numpy(Z).float())
    # Y_conv = ys.detach().cpu().numpy().reshape(Z1.shape)
    #
    # Xhat = model.decode(torch.from_numpy(np.concatenate([np.ravel(Z1).reshape(-1, 1),
    #                                                     np.ravel(Z2).reshape(-1, 1)],
    #                                                    axis=1)).float().cuda()).detach().cpu().numpy()
    # yhat =np.apply_along_axis( f, axis=1, arr=Xhat )
    # Yhat = yhat.reshape(Z1.shape)
    # ax.plot_surface(Z1, Z2, Yhat)
    # ax.plot_surface(Z1, Z2, Y_conv)
    # ax.set_zlim(0, 5)




