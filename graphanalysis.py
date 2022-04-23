# This code was written by Kristoffer Hougaard Madsen

import numpy as np
import scipy
import torch
from torch import nn, distributions
from scipy.spatial.distance import cdist
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statistics import covariance
import pandas as pd
import seaborn as sns
from bayesianfunctions import getGP_plot

torch.set_default_tensor_type(torch.FloatTensor)

def load(filename):
    df = pd.read_csv('data/'+str(filename)).iloc[:,1:]
    data = df.to_numpy()
    return data, df
    

def mean(A):
    return np.mean(A, axis=0)
def var(A):
    return np.var(A, axis=0)
def corM(df, method):
    return df.corr(method=method)

def ttest(vec1, vec2):
    return scipy.stats.ttest_rec(vec1,vec2)[1]

a, df = load('data_98_observational.csv')
nodes = df.columns

means = mean(a)
vars = var(a)

#corM(df, 'pearson')

#corM(df, 'spearman')

#df.describe()

sns.set(style="white")
sns.pairplot(df, kind="scatter",diag_kind="kde")
plt.show()


    