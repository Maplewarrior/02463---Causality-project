# This code was written by Kristoffer Hougaard Madsen

import numpy as np
import scipy
import torch
from torch import nn, distributions
from scipy.spatial.distance import cdist
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statistics import covariance, mean
import pandas as pd
import seaborn as sns
from bayesianfunctions import getGP_plot

torch.set_default_tensor_type(torch.FloatTensor)

def corM(df, method):
    return df.corr(method=method)

def ttest(vec1, vec2):
    return scipy.stats.ttest_rec(vec1,vec2)[1]

class CausalityAnalysis:
    def __init__(self,intervention):
        self.data, self.df = self.load(self.getFilename(intervention))
    
    def load(self,filename):
        df = pd.read_csv('data/'+str(filename)).iloc[:,1:]
        data = df.to_numpy()
        return data, df

    def getFilename(self,intervention):
        return "data_" + intervention + ".csv"
    
    def getVariableMeans(self):
        return np.mean(self.data, axis=0)

    def getVariableVariances(self):
        return np.var(self.data, axis=0)

    def getScatterplots(self):
        sns.set(style="white")
        sns.pairplot(self.df, kind="scatter",diag_kind="kde")
        plt.show()

    def getDescriptiveStatistics(self):
        print(self.df.describe())

interventions = ["98_observational","30_A0","30_B0.16","30_C0","30_D0","30_E0","50_F-0.5"]

mean_df = pd.DataFrame(np.zeros((len(interventions),6)))
    
mean_df.index = interventions

mean_df.columns = ["A","B","C","D","E","F"]

for intervention in interventions:
    ca = CausalityAnalysis(intervention)
    mean_df.loc[intervention,:] = ca.getVariableMeans()

print(mean_df)

var_df = pd.DataFrame(np.zeros((len(interventions),6)))
    
var_df.index = interventions

var_df.columns = ["A","B","C","D","E","F"]

for intervention in interventions:
    ca = CausalityAnalysis(intervention)
    var_df.loc[intervention,:] = ca.getVariableVariances()

print(var_df)


for intervention in interventions:
    ca = CausalityAnalysis(intervention)
    ca.getScatterplots()

#corM(df, 'spearman')






    