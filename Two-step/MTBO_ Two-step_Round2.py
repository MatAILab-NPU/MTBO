#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.initial_designs.random_design import RandomDesign
from chimera import Chimera
from emukit.core.loop.user_function import UserFunctionWrapper
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import plotly.express as px
import plotly.graph_objects as go
from numpy.linalg import norm
import pygwalker as pyg
import matplotlib.ticker as ticker
import os
import scipy.stats
import math
from IPython.display import SVG, display
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.sans-serif'] = ["Arial"]
warnings.filterwarnings('ignore')


# In[2]:


R1_min,R1_max, R1_step = [500, 4500, 500] ## Unit: rpm, 9 steps
R1_var = np.arange(R1_min, R1_max+R1_step*0.1, R1_step)
R1_num = len(R1_var)

T_min, T_max, T_step = [50, 100, 5] ## Unit: ℃, # 11 steps
T_var = np.arange(T_min, T_max+T_step*0.1, T_step)
T_num = len(T_var)

R2_min,R2_max, R2_step = [500, 4500, 500] ## Unit: rpm, 9 steps
R2_var = np.arange(R2_min, R2_max+R2_step*0.1, R2_step)
R2_num = len(R2_var)

H_min, H_max, H_step = [20, 80, 10] ## Unit: %, # 7 steps
H_var = np.arange(H_min, H_max+H_step*0.1, H_step)
H_num = len(H_var)

MACl_min, MACl_max, MACl_step = [0, 50, 5] ## Unit: %, 11 steps
MACl_var = np.arange(MACl_min, MACl_max+MACl_step*0.1, MACl_step)
MACl_num = len(MACl_var)

HT_min, HT_max, HT_step = [0.2, 2, 0.2] ## Unit: %, 10 steps
HT_var = np.arange(HT_min, HT_max+HT_step*0.1,HT_step)
HT_num = len(HT_var)


var_array = [R1_var,T_var,R2_var, H_var, MACl_var,HT_var]

x_labels = ['R1[rpm]', 
            'T [degC]',
            'R2[rpm]',
            'Humidity[%]',
            'MACl [%]',
            'HT[%]']    


# In[3]:


def x_normalizer(X, var_array = var_array):
    
    def max_min_scaler(x, x_max, x_min):
        return (x-x_min)/(x_max-x_min)
    x_norm = []
    for x in (X):
           x_norm.append([max_min_scaler(x[i], 
                         max(var_array[i]), 
                         min(var_array[i])) for i in range(len(x))])
            
    return np.array(x_norm)

def x_denormalizer(x_norm, var_array = var_array):
    
    def max_min_rescaler(x, x_max, x_min):
        return x*(x_max-x_min)+x_min
    x_original = []
    for x in (x_norm):
           x_original.append([max_min_rescaler(x[i], 
                              max(var_array[i]), 
                              min(var_array[i])) for i in range(len(x))])

    return np.array(x_original)

def get_closest_value(given_value, array_list):
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    closest_value = min(array_list, key=absolute_difference_function)
    return closest_value
    
def get_closest_array(suggested_x, var_list):
    modified_array = []
    for x in suggested_x:
        modified_array.append([get_closest_value(x[i], var_list[i]) for i in range(len(x))])
    return np.array(modified_array)


# In[4]:


from typing import Tuple, Union
import scipy.stats
import numpy as np
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel, IDifferentiable

class ScaledProbabilityOfFeasibility(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], jitter: float = float(0),
                 max_value: float = float(1), min_value: float = float(0)) -> None:
        """
        This acquisition computes for a given input point the probability of satisfying the constraint
        C<0. For more information see:
        Michael A. Gelbart, Jasper Snoek, and Ryan P. Adams,
        Bayesian Optimization with Unknown Constraints,
        https://arxiv.org/pdf/1403.5607.pdf
        :param model: The underlying model that provides the predictive mean and variance for the given test points
        :param jitter: Jitter to balance exploration / exploitation
        """
        self.model = model
        self.jitter = jitter
        self.max_value = max_value
        self.min_value = min_value

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the probability of of satisfying the constraint C<0.
        :param x: points where the acquisition is evaluated, shape (number of points, number of dimensions).
        :return: numpy array with the probability of satisfying the constraint at the points x.
        """
        mean, variance = self.model.predict(x)
        mean += self.jitter

        standard_deviation = np.sqrt(variance)
        cdf = scipy.stats.norm.cdf(0, mean, standard_deviation)
        return cdf*(self.max_value-self.min_value)+self.min_value

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the  probability of of satisfying the constraint C<0.
        :param x: points where the acquisition is evaluated, shape (number of points, number of dimensions).
        :return: tuple of numpy arrays with the probability of satisfying the constraint at the points x 
        and its gradient.
        """
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_devidation_dx = dvariance_dx / (2 * standard_deviation)

        mean += self.jitter
        u = - mean / standard_deviation
        pdf = scipy.stats.norm.pdf(0, mean, standard_deviation)
        cdf = scipy.stats.norm.cdf(0, mean, standard_deviation)
        dcdf_dx = - pdf * (dmean_dx + dstandard_devidation_dx * u)

        return cdf*(self.max_value-self.min_value)+self.min_value, dcdf_dx

    @property
    def has_gradients(self):
        return isinstance(self.model, IDifferentiable)


# In[5]:


X_all_grid = []
for R1 in R1_var:
    for T in T_var:
        for R2 in R2_var:
            for H in H_var:
                for MACl in MACl_var:
                    for HT in HT_var:
                        X_all_grid.append([R1, T, R2, H, MACl,HT])
X_all_grid = np.array(X_all_grid)
X_all_grid.shape


# In[6]:


x_eva=x_normalizer(X_all_grid)


# In[7]:


### Add/minus a half step to make sure the edge conditions have the same chance in nearest neighbors
parameter_space = ParameterSpace([ContinuousParameter('x1', 0-1/(R1_num-1)/2, 1+1/(R1_num-1)/2),
                                  ContinuousParameter('x2', 0-1/(T_num-1)/2,  1+1/(T_num-1)/2),
                                  ContinuousParameter('x3', 0-1/(R2_num-1)/2,    1+1/(R2_num-1)/2),
                                  ContinuousParameter('x4', 0-1/(H_num-1)/2,    1+1/(H_num-1)/2),
                                  ContinuousParameter('x5', 0-1/(MACl_num-1)/2,    1+1/(MACl_num-1)/2),
                                  ContinuousParameter('x6', 0-1/(HT_num-1)/2,    1+1/(HT_num-1)/2)])


# In[8]:


df_film = pd.read_csv('./data_film.csv')
df_film


# In[9]:


df_exp1 = pd.read_csv('./data_BG.csv')
success_conditions = df_exp1[df_exp1['Success or Fail']==1]['ML Condition'].values
df_obj1 = df_exp1[df_exp1['ML Condition'].isin(success_conditions)]
df_obj1


# In[10]:


df_exp2 = pd.read_csv('./data_PL.csv')
success_conditions = df_exp2[df_exp2['Success or Fail']==1]['ML Condition'].values
df_obj2 = df_exp2[df_exp2['ML Condition'].isin(success_conditions)]
df_obj2


# In[11]:


x1=df_obj1.iloc[:,1:7].values
x2=df_obj2.iloc[:,1:7].values
x3=df_film.iloc[:,1:7].values
x1=x_normalizer(x1)
x2=x_normalizer(x2)
x3=x_normalizer(x3)
y1=np.transpose([df_obj1.iloc[:,-2].values])#△bandgap
y1=abs(y1)
y2=np.transpose([df_obj2.iloc[:,-2].values])#△peak
y2=abs(y2)
y3=np.transpose([df_film.iloc[:,-1].values])#Film quality


# In[12]:


import GPy
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
X1, Y1 = [x1, y1]
X2, Y2 = [x2, y2]
X3, Y3 = [x3, y3]

input_dim = len(X1[0])
ker1 = GPy.kern.Matern52(input_dim = input_dim, ARD =True)
ker1.lengthscale.constrain_bounded(1e-2, 10)
ker1.variance.constrain_bounded(1e-2, 1e5)
model1_gpy = GPRegression(X1, Y1, ker1)
model1_gpy.Gaussian_noise.variance = 2**2
model1_gpy.Gaussian_noise.variance.fix()
model1_gpy.randomize()
model1_gpy.optimize_restarts(num_restarts=20,verbose =False, messages=False)
objective_model1 = GPyModelWrapper(model1_gpy)
print(objective_model1.model.kern.lengthscale)
print(objective_model1.model.kern.variance)

input_dim = len(X2[0])
ker2 = GPy.kern.Matern52(input_dim = input_dim, ARD =True)
ker2.lengthscale.constrain_bounded(1e-2, 10)
ker2.variance.constrain_bounded(1e-2, 1e5)
model2_gpy = GPRegression(X2, Y2, ker2)
model2_gpy.Gaussian_noise.variance = 2**2
model2_gpy.Gaussian_noise.variance.fix()
model2_gpy.randomize()
model2_gpy.optimize_restarts(num_restarts=20,verbose =False, messages=False)
objective_model2 = GPyModelWrapper(model2_gpy)
print(objective_model2.model.kern.lengthscale)
print(objective_model2.model.kern.variance)

input_dim = len(X3[0])
ker3 = GPy.kern.RBF(input_dim = input_dim, ARD =True)
ker3.lengthscale.constrain_bounded(1e-1, 1)
ker3.variance.constrain_bounded(1e-1, 1e3)
# ker3 += GPy.kern.White(input_dim = input_dim)
yc_offset = 0.75
model3_gpy = GPRegression(X3, -(Y3-yc_offset), ker3)
# model3_gpy.Gaussian_noise.variance = 0.1**2
# model3_gpy.Gaussian_noise.variance.fix()
model3_gpy.randomize()
model3_gpy.optimize(max_iters=1000, messages=False)  # Use a fixed number of iterations
model3_gpy.optimize_restarts(num_restarts=20, verbose=False)  # Reduce the number of restarts
objective_model3 = GPyModelWrapper(model3_gpy)
print(objective_model3.model.kern.lengthscale)
print(objective_model3.model.kern.variance)

f_obj1 = objective_model1.model.predict
f_obj2 = objective_model2.model.predict
f_obj3 = objective_model3.model.predict

y1_pred, y1_uncer = f_obj1(X1)
y1_pred = y1_pred[:,-1]
y2_pred, y2_uncer = f_obj2(X2)
y2_pred = y2_pred[:,-1]
y3_pred, y3_uncer = f_obj3(X3)
y3_pred = -y3_pred[:,-1]+yc_offset

y1_uncer = np.sqrt(y1_uncer[:,-1])
y2_uncer = np.sqrt(y2_uncer[:,-1])
y3_uncer = np.sqrt(y3_uncer[:,-1])


# In[14]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fs = 18

lims1 = (-.1, 70.1)
axes[0].scatter(Y1[:,-1], y1_pred, alpha=0.5, edgecolor='white', c='royalblue')
axes[0].errorbar(Y1[:,-1], y1_pred, yerr=y1_uncer, ms=0, ls='', capsize=2, alpha=0.5, color='gray', zorder=0)
axes[0].plot(lims1, lims1, 'k--', alpha=0.75, zorder=0)
rmse_value = np.sqrt(mean_squared_error(Y1[:,-1], y1_pred))
title = 'GPR for Delta_Bandgap' + " (RMSE=%.2f" % rmse_value + ')'
axes[0].set_xlabel('Ground Truth', fontsize=fs)
axes[0].set_ylabel('Prediction', fontsize=fs)
axes[0].set_title(title, fontsize=fs)
axes[0].tick_params(axis='both', labelsize=14, direction='in')  # 设置刻度字体大小和方向
mse_peak = mean_squared_error(Y1[:,-1], y1_pred)
print('Bandgap rmse: %.4f' % (np.sqrt(mse_peak)))
rsquared_peak = r2_score(Y1[:,-1], y1_pred)
print('Bandgap R^2: %.4f' % rsquared_peak)
sprman_peak = spearmanr(Y1[:,-1], y1_pred)
print('Bandgap spearman: %.4f' % sprman_peak[0])

lims2 = (-.1, 70.1)
axes[1].scatter(Y2[:,-1], y2_pred, alpha=0.5, edgecolor='white', c='royalblue')
axes[1].errorbar(Y2[:,-1], y2_pred, yerr=y2_uncer, ms=0, ls='', capsize=2, alpha=0.5, color='gray', zorder=0)
axes[1].plot(lims2, lims2, 'k--', alpha=0.75, zorder=0)
rmse_value = np.sqrt(mean_squared_error(Y2[:,-1], y2_pred))
title = 'GPR for Delta_Peak' + " (RMSE=%.2f" % rmse_value + ')'
axes[1].set_xlabel('Ground Truth', fontsize=fs)
axes[1].set_ylabel('Prediction', fontsize=fs)
axes[1].set_title(title, fontsize=fs)
axes[1].tick_params(axis='both', labelsize=14, direction='in')
mse_trpl = mean_squared_error(Y2[:,-1], y2_pred)
print('Peak rmse: %.4f' % (np.sqrt(mse_trpl)))
rsquared_trpl = r2_score(Y2[:,-1], y2_pred)
print('Peak R^2: %.4f' % rsquared_trpl)
sprman_trpl = spearmanr(Y2[:,-1], y2_pred)
print('Peak spearman: %.4f' % sprman_trpl[0])

lims3 = (-.1, 1.1)
axes[2].scatter(Y3[:,-1], y3_pred, alpha=0.5, edgecolor='white', c='royalblue')
axes[2].errorbar(Y3[:,-1], y3_pred, yerr=y3_uncer, ms=0, ls='', capsize=2, alpha=0.5, color='gray', zorder=0)
axes[2].plot(lims3, lims3, 'k--', alpha=0.75, zorder=0)
rmse_value = np.sqrt(mean_squared_error(Y3[:,-1], y3_pred))
title = 'GPR for Binary Film Quality'
axes[2].set_xlabel('Ground Truth', fontsize=fs)
axes[2].set_ylabel('Prediction', fontsize=fs)
axes[2].set_title(title, fontsize=fs)
axes[2].tick_params(axis='both', labelsize=14, direction='in')
mse_film = mean_squared_error(Y3[:,-1], y3_pred)
print('Film rmse: %.4f' % (np.sqrt(mse_film)))
rsquared_film = r2_score(Y3[:,-1], y3_pred)
print('Film R^2: %.4f' % rsquared_film)
sprman_film = spearmanr(Y3[:,-1], y3_pred)
print('Film spearman: %.4f' % sprman_film[0])

plt.subplots_adjust(wspace=0.4)
plt.show()


# In[15]:


from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound,ProbabilityOfFeasibility,ProbabilityOfImprovement
acquisition1 = NegativeLowerConfidenceBound(objective_model1,beta = 1)
acquisition2 = NegativeLowerConfidenceBound(objective_model2,beta = 1)
acquisition_constraint = ProbabilityOfFeasibility(objective_model3)
obj1_plot = -acquisition1.evaluate(x_eva)#NLCB of obj1
obj2_plot = -acquisition2.evaluate(x_eva)#LCB of obj2
cons_plot = acquisition_constraint.evaluate(x_eva)

tolerances = np.array([0.2,0.4]) 
absolutes = [False, False]
goals = ['min', 'min'] 
chimera = Chimera(tolerances=tolerances, absolutes=absolutes, goals=goals)
obj = np.array([obj1_plot.T[0], obj2_plot.T[0]])
scalarized = chimera.scalarize(obj.T)
cons_acq = (1-scalarized)*cons_plot.T[0]#constrained acquisition
np.max(cons_acq)


# In[442]:


np.random.seed(42)  
bs = 15
top = 686  # top .1% = 686/686070
sort_index = np.argsort(cons_acq, axis=0)

X_new = []

while len(X_new) < bs:
    X_candidates = [X_all_grid[i] for i in sort_index[-top:]]
    X_candidates = np.array(X_candidates)
    X_filtered = X_candidates[X_candidates[:, 4] < 45]  
    if len(X_filtered) >= bs:
        bs_index = np.random.choice(len(X_filtered), size=bs, replace=False) 
        X_new = X_filtered[bs_index]
    else:
        top += 100 

X_new = np.array(X_new)

cons_pr = acquisition_constraint.evaluate(x_normalizer(X_new))

idx = sort_index[-top:][bs_index]

final_acq = cons_acq[idx]

y_new_pred1, y_new_uncer1 = f_obj1(x_normalizer(X_new))
y_new_pred2, y_new_uncer2 = f_obj2(x_normalizer(X_new))

df_Xnew = pd.DataFrame(X_new, columns=x_labels)

df_all = pd.concat([df_film.iloc[:, 1:7], df_Xnew])

df_all_ = df_all.drop_duplicates()

df_Xnew = df_all_.iloc[len(df_film):len(df_film) + bs]

df_Xnew = df_Xnew.sort_values(by=list(df_film.columns[1:7]), ignore_index=True)

df_Xnew.index = np.arange(len(df_Xnew)) + len(df_film)

print('New X:', len(df_Xnew))

print('Final top value used:', top)

df_Xnew


# In[460]:


df_Xnew.to_csv('2nd round suggestions.csv')


# In[443]:


df_x = df_Xnew
df_cols = df_film.columns
n_col = 3 # num of columns per row in the figure
fs = 24
for n in np.arange(0, 6, n_col):
    fig,axes = plt.subplots(1, n_col, figsize=(10, 3), sharey = False)
    fs = 24
    for i in np.arange(n_col):
        if n< len(df_cols):
            axes[i].hist(df_x.iloc[:,n], bins= 20, range = (min(var_array[n])- 0.05*abs(var_array[n][1]-var_array[n][0]),
                                                            max(var_array[n])+0.05*abs(var_array[n][1]-var_array[n][0])), 
                         edgecolor='black')
            axes[i].set_xlabel(df_cols[n+1], fontsize = 18)


        else:
            axes[i].axis("off")
        n = n+1      
    axes[0].set_ylabel('counts', fontsize = fs)
    for i in range(len(axes)):
        axes[i].tick_params(direction='in', length=5, width=1, labelsize = fs*.8, grid_alpha = 0.5)
        axes[i].grid(True, linestyle='-.')
    plt.show()


# In[17]:


fig, axes = plt.subplots(3, 1, figsize=(8, 18), sharey = False, sharex = False)
fs = 20

film_life = df_exp1.sort_values('ML Condition').iloc[:,-2].values
exp_life = film_life.reshape(-1,1)

data_life = df_obj1.sort_values('ML Condition').iloc[:,[0,-2]].values
exp_cond = data_life[:,0]

f_obj = objective_model1.model.predict
y_pred, y_uncer = f_obj(X1)
y_pred = y_pred[:,-1]
y_uncer = np.sqrt(y_uncer[:,-1])

unsuccess_idx=df_exp1[df_exp1['Success or Fail']==0]['ML Condition'].values
unsuccess_film=df_exp1[df_exp1['Success or Fail']==0]['Success or Fail'].values
unsuccess_film=unsuccess_film
all_cond = np.concatenate([data_life[:,0], np.transpose(unsuccess_idx)])
all_cond = all_cond[np.argsort(all_cond)]

axes[0].scatter(unsuccess_idx, unsuccess_film,facecolor = 'black',edgecolor = 'black',s = 20, label = 'failed')
axes[0].scatter(exp_cond, y1, facecolor = 'none',edgecolor = 'navy', s = 20, alpha = 0.6, label = 'experiment')
axes[0].scatter(exp_cond, y_pred,s = 50, facecolors='none', alpha = 0.6, edgecolor = 'gray', label = 'predicted')
axes[0].errorbar(exp_cond, y_pred, yerr = y_uncer,   ms = 1, ls = '', capsize = 2, alpha = 0.6, color = 'gray', zorder = 0)
axes[0].plot(exp_cond, np.minimum.accumulate(y1), marker = 'o', ms = 0, c = 'black')

y_pred_new, y_uncer_new = f_obj(x_normalizer(df_Xnew.values))
y_pred_new = y_pred_new[:,-1]
y_uncer_new = np.sqrt(y_uncer_new[:,-1])
axes[0].scatter(np.arange(len(df_Xnew))+len(exp_life), y_pred_new,s = 50, facecolors='none', alpha = 0.6, edgecolor = 'darkgreen', label = 'suggested')
axes[0].errorbar(np.arange(len(df_Xnew))+len(exp_life), y_pred_new, yerr = y_uncer_new,  ms = 0, ls = '', capsize = 2, alpha = 0.6, color = 'darkgreen', zorder = 0)

axes[0].set_ylabel('Current Best delta_bandgap', fontsize = 20)
axes[0].set_xlabel('Process Condition', fontsize = 20)

axes[0].set_ylim(-25, 70)
axes[0].set_xlim(-1, 65)
axes[0].set_xticks(np.arange(0,65,5))
axes[0].legend(fontsize = fs*0.7)


film_uniform = df_exp2.sort_values('ML Condition').iloc[:,-2].values
exp_uniform = film_uniform.reshape(-1,1)

data_uniform = df_obj2.sort_values('ML Condition').iloc[:,[0,-2]].values
exp_cond = data_uniform[:,0]

f_obj = objective_model2.model.predict
y_pred, y_uncer = f_obj(X2)
y_pred = y_pred[:,-1]
y_uncer = np.sqrt(y_uncer[:,-1])

unsuccess_idx=df_exp2[df_exp2['Success or Fail']==0]['ML Condition'].values
unsuccess_film=df_exp2[df_exp2['Success or Fail']==0]['Success or Fail'].values
unsuccess_film=unsuccess_film
all_cond = np.concatenate([data_uniform[:,0], np.transpose(unsuccess_idx)])
all_cond = all_cond[np.argsort(all_cond)]

axes[1].scatter(unsuccess_idx, unsuccess_film,facecolor = 'black',edgecolor = 'black',s = 20, label = 'failed')
axes[1].scatter(exp_cond, y2, facecolor = 'none',edgecolor = 'navy', s = 20, alpha = 0.6, label = 'experiment')
axes[1].scatter(exp_cond, y_pred,s = 50, facecolors='none', alpha = 0.6, edgecolor = 'gray', label = 'predicted')
axes[1].errorbar(exp_cond, y_pred, yerr = y_uncer,  ms = 1, ls = '', capsize = 2, alpha = 0.6, color = 'gray', zorder = 0)
axes[1].plot(exp_cond, np.minimum.accumulate(y2), marker = 'o', ms = 0, c = 'black')

y_pred_new, y_uncer_new = f_obj(x_normalizer(df_Xnew.values))
y_pred_new = y_pred_new[:,-1]
y_uncer_new = np.sqrt(y_uncer_new[:,-1])

axes[1].scatter(np.arange(len(df_Xnew))+len(exp_uniform), y_pred_new,s = 50, facecolors='none', alpha = 0.6, edgecolor = 'darkgreen', label = 'suggested')
axes[1].errorbar(np.arange(len(df_Xnew))+len(exp_uniform), y_pred_new, yerr = y_uncer_new,  ms = 0, ls = '', capsize = 2, alpha = 0.6, color = 'darkgreen', zorder = 0)

axes[1].set_ylabel('Current Best delta_peak', fontsize = 20)
axes[1].set_xlabel('Process Condition', fontsize = 20)
axes[1].set_ylim(-20, 100)
axes[1].set_xlim(-1, 65)
axes[1].set_xticks(np.arange(0,65,5))
axes[1].legend(fontsize = fs*0.7,loc=1)


film_quality = df_film.sort_values('ML Condition').iloc[:,-1].values
exp_quality = film_quality.reshape(-1,1)

data_quality = df_film.sort_values('ML Condition').iloc[:,[0,-3]].values
exp_cond = data_quality[:,0]

f_obj = objective_model3.model.predict
y_pred, y_uncer = f_obj(X3)
y_pred = -y_pred[:,-1]+yc_offset
y_uncer = np.sqrt(y_uncer[:,-1])

unsuccess_idx=df_film[df_film['Film Quality']==0]['ML Condition'].values
unsuccess_film=df_film[df_film['Film Quality']==0]['Film Quality'].values

success_idx=df_film[df_film['Film Quality']==1]['ML Condition'].values
success_film=df_film[df_film['Film Quality']==1]['Film Quality'].values
# all_cond = np.concatenate([data_quality[:,0], np.transpose(unsuccess_idx)])
# all_cond = all_cond[np.argsort(all_cond)]

axes[2].scatter(unsuccess_idx, unsuccess_film,facecolor = 'black',edgecolor = 'black',s = 20, label = 'failed')
axes[2].scatter(success_idx, success_film, facecolor = 'none',edgecolor = 'navy', s = 20, alpha = 0.6, label = 'experiment')
y_pred_new, y_uncer_new = f_obj(x_normalizer(df_Xnew.values))
y_pred_new = -y_pred_new[:,-1]+yc_offset
y_uncer_new = np.sqrt(y_uncer_new[:,-1])

axes[2].scatter(np.arange(len(df_Xnew))+len(exp_quality), y_pred_new,s = 50, facecolors='none', alpha = 0.6, edgecolor = 'darkgreen', label = 'suggested')
axes[2].errorbar(np.arange(len(df_Xnew))+len(exp_quality), y_pred_new, yerr = y_uncer_new,  ms = 0, ls = '', capsize = 2, alpha = 0.6, color = 'darkgreen', zorder = 0)

axes[2].set_ylabel('Current Best Film', fontsize = 20)
axes[2].set_xlabel('Process Condition', fontsize = 20)

axes[2].set_ylim(-.1, 1.3)
axes[2].set_xlim(-1, 65)
axes[2].set_xticks(np.arange(0,65,5))
axes[2].legend(fontsize = fs*0.7)

for ax in axes:
    ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8, grid_alpha = 0.5)
    ax.grid(True, linestyle='-.')
    ax.legend(ncol=2)
plt.subplots_adjust(wspace = 0.8)

plt.show()


# In[446]:


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey = False)
fs = 20

# axes[0].plot(np.arange(len(df_Xnew))+1+len(X2), acq_new1, marker = 'o',
#                 ms = 2, alpha = 0.6, color = 'orange', label = 'raw acqui1')
# axes[0].plot(np.arange(len(df_Xnew))+1+len(X2), acq_new1, marker = 'o',
#                 ms = 2, alpha = 0.6, color = 'navy', label = 'raw acqui2')
axes[0].plot(np.arange(len(df_Xnew))+len(df_film), cons_pr, marker = 'o',
                ms = 2, alpha = 0.6, color = 'red', label = 'constr prob')
axes[0].plot(np.arange(len(df_Xnew))+len(df_film), final_acq, marker = 'o',
                ms = 2, alpha = 0.6, color = 'darkgreen', label = 'final acqui')

axes[0].set_ylim(0., 1)
axes[0].set_xlim(-1, 65)
axes[0].set_xticks(np.arange(0,65,10))
axes[0].set_ylabel('Acquisition Probability', fontsize = fs)
axes[0].set_xlabel('Process Condition', fontsize = fs)
axes[0].legend(fontsize = fs*0.7,loc=4)

axes[1].axis("off")
for ax in axes:
    ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8, grid_alpha = 0.5)
    ax.grid(True, linestyle='-.')
plt.subplots_adjust(wspace = 0.4)

plt.show()


# In[447]:


design = RandomDesign(parameter_space)
x_sampled = design.get_samples(200)
x_sampled = x_sampled
input_dim = 6
for i in range(input_dim):
    for j in range(input_dim-i-1):
        ind1 = i
        ind2 = j+i+1
        n_steps =21
        x1x2y_pred, x1x2y_uncer =[[],[]]
        for x1 in np.linspace(0, 1, n_steps):
            for x2 in np.linspace(0, 1, n_steps):
                x_temp = np.copy(x_sampled)
                x_temp[:,ind1] = x1
                x_temp[:,ind2] = x2
                y_pred, y_uncer = f_obj3(x_temp)
                y2 = -y_pred+yc_offset
                x1_org = x_denormalizer(x_temp)[0,ind1]
                x2_org = x_denormalizer(x_temp)[0,ind2]
                x1x2y_pred.append([x1_org, x2_org, np.max(y2), np.mean(y2), np.min(y2)])
                x1x2y_uncer.append([x1_org, x2_org, np.max(np.sqrt(y_uncer)), np.mean(np.sqrt(y_uncer)), np.min(np.sqrt(y_uncer))])
        
        x1 = np.array(x1x2y_pred, dtype=object)[:,0].reshape(n_steps, n_steps)
        x2 = np.array(x1x2y_pred, dtype=object)[:,1].reshape(n_steps, n_steps)
            
        y_max2 = np.array(x1x2y_pred, dtype=object)[:,2].reshape(n_steps, n_steps)
        y_mean2 = np.array(x1x2y_pred, dtype=object)[:,3].reshape(n_steps, n_steps)
        y_min2 = np.array(x1x2y_pred, dtype=object)[:,4].reshape(n_steps, n_steps)
        
        y_uncer_max = np.array(x1x2y_uncer, dtype=object)[:,2].reshape(n_steps, n_steps)
        y_uncer_mean = np.array(x1x2y_uncer, dtype=object)[:,3].reshape(n_steps, n_steps)
        y_uncer_min = np.array(x1x2y_uncer, dtype=object)[:,4].reshape(n_steps, n_steps)

        fs = 20
        title_pad = 16
        
        fig,axes = plt.subplots(1, 3, figsize=(17, 4), sharey = False, sharex = False)
#         for ax, y in zip(axes,
#                    [y_max2, y_mean2, y_min2]):
#             c_plt1 = ax.contourf(x1, x2, y,cmap='viridis',extend='both')
        colorbar_offset = [0.5, 0.3, 0]
        for ax, c_offset, y in zip(axes,colorbar_offset,
                           [y_max2, y_mean2, y_min2]):
            c_plt1 = ax.contourf(x1, x2, y,levels = np.arange(20)/2*0.05+c_offset,cmap='viridis',extend = 'both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X3)[:, ind1], 
                       x_denormalizer(X3)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
            ax.scatter((X_new)[:, ind1], 
                       (X_new)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'green')
#             axes[0].contour(x1, x2, y_min2, colors='r',levels=[yc2_offset])
            
            ax.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
            ax.set_ylabel(str(x_labels[ind2]),fontsize =  fs)
            
            x1_delta = (np.max(x1)-np.min(x1))*0.05
            x2_delta = (np.max(x2)-np.min(x2))*0.05
            ax.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
            ax.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
            
            ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
            if ind1==0:#R1
                    ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==1:#T
                ax.set_xticks([50, 60, 70, 80, 90, 100])
            if ind1==2:#R2
                ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==3:#H
                ax.set_xticks([20, 40, 60, 80])
            if ind1==4:#MACl%
                ax.set_xticks([0, 10, 20, 30, 40, 50])
            if ind1==5:#HT%
                ax.set_yticks([0.2, 0.8, 1.4, 2.0])
                
        axes[0].set_title('film quality max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('film quality mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('film quality min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()


# In[489]:


design = RandomDesign(parameter_space)
x_sampled = design.get_samples(200)
x_sampled = x_sampled
input_dim = 6
for i in range(input_dim):
    for j in range(input_dim-i-1):
        ind1 = i
        ind2 = j+i+1
        n_steps =21
        x1x2y_pred, x1x2y_uncer =[[],[]]
        for x1 in np.linspace(0, 1, n_steps):
            for x2 in np.linspace(0, 1, n_steps):
                x_temp = np.copy(x_sampled)
                x_temp[:,ind1] = x1
                x_temp[:,ind2] = x2
                y_pred, y_uncer = f_obj1(x_temp)
                y2 = y_pred
                x1_org = x_denormalizer(x_temp)[0,ind1]
                x2_org = x_denormalizer(x_temp)[0,ind2]
                x1x2y_pred.append([x1_org, x2_org, np.max(y2), np.mean(y2), np.min(y2)])
                x1x2y_uncer.append([x1_org, x2_org, np.max(np.sqrt(y_uncer)), np.mean(np.sqrt(y_uncer)), np.min(np.sqrt(y_uncer))])
        
        x1 = np.array(x1x2y_pred, dtype=object)[:,0].reshape(n_steps, n_steps)
        x2 = np.array(x1x2y_pred, dtype=object)[:,1].reshape(n_steps, n_steps)
            
        y_max2 = np.array(x1x2y_pred, dtype=object)[:,2].reshape(n_steps, n_steps)
        y_mean2 = np.array(x1x2y_pred, dtype=object)[:,3].reshape(n_steps, n_steps)
        y_min2 = np.array(x1x2y_pred, dtype=object)[:,4].reshape(n_steps, n_steps)
        
        y_uncer_max = np.array(x1x2y_uncer, dtype=object)[:,2].reshape(n_steps, n_steps)
        y_uncer_mean = np.array(x1x2y_uncer, dtype=object)[:,3].reshape(n_steps, n_steps)
        y_uncer_min = np.array(x1x2y_uncer, dtype=object)[:,4].reshape(n_steps, n_steps)

        fs = 20
        title_pad = 16
        
        fig,axes = plt.subplots(1, 3, figsize=(17, 4), sharey = False, sharex = False)
#         for ax, y in zip(axes,
#                            [y_max2, y_mean2, y_min2]):
#             c_plt1 = ax.contourf(x1, x2, y,cmap='plasma',extend='both')
        colorbar_offset = [40, 30, -10]
        for ax, c_offset, y in zip(axes,colorbar_offset,
                           [y_max2, y_mean2, y_min2]):
            c_plt1 = ax.contourf(x1, x2, y,levels = np.arange(21)+c_offset,cmap='plasma',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            # 修改 colorbar 刻度值显示（加 10）
            cbar_ticks = np.array([tick + 10 for tick in cbar.get_ticks()])
            cbar.set_ticks(cbar_ticks - 10)  # 将实际数据映射回原值
            cbar.set_ticklabels(cbar_ticks)  # 显示加 10 后的标签
            cbar.ax.tick_params(labelsize=fs*0.8)     
            ax.scatter(x_denormalizer(X2)[:, ind1], 
                       x_denormalizer(X2)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
            ax.scatter((X_new)[:, ind1], 
                       (X_new)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'green')
#             axes[0].contour(x1, x2, y_max2, colors='tan',levels=[150])
#             axes[0].contour(x1, x2, y_max2, colors='tan',levels=[0.4*(np.max(y_max2)-np.min(y_max2))+np.min(y_max2)])
            
            ax.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
            ax.set_ylabel(str(x_labels[ind2]),fontsize =  fs)
            
            x1_delta = (np.max(x1)-np.min(x1))*0.05
            x2_delta = (np.max(x2)-np.min(x2))*0.05
            ax.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
            ax.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
            
            ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
            if ind1==0:#R1
                    ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==1:#T
                ax.set_xticks([50, 60, 70, 80, 90, 100])
            if ind1==2:#R2
                ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==3:#H
                ax.set_xticks([20, 40, 60, 80])
            if ind1==4:#MACl%
                ax.set_xticks([0, 10, 20, 30, 40, 50])
            if ind1==5:#HT%
                ax.set_yticks([0.2, 0.8, 1.4, 2.0])
                
        axes[0].set_title('Delta_Bandgap max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('Delta_Bandgap mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('Delta_Bandgap min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()


# In[476]:


design = RandomDesign(parameter_space)
x_sampled = design.get_samples(200)
x_sampled = x_sampled
input_dim = 6
for i in range(input_dim):
    for j in range(input_dim-i-1):
        ind1 = i
        ind2 = j+i+1
        n_steps =21
        x1x2y_pred, x1x2y_uncer =[[],[]]
        for x1 in np.linspace(0, 1, n_steps):
            for x2 in np.linspace(0, 1, n_steps):
                x_temp = np.copy(x_sampled)
                x_temp[:,ind1] = x1
                x_temp[:,ind2] = x2
                y_pred, y_uncer = f_obj2(x_temp)
                y2 = y_pred
                x1_org = x_denormalizer(x_temp)[0,ind1]
                x2_org = x_denormalizer(x_temp)[0,ind2]
                x1x2y_pred.append([x1_org, x2_org, np.max(y2), np.mean(y2), np.min(y2)])
                x1x2y_uncer.append([x1_org, x2_org, np.max(np.sqrt(y_uncer)), np.mean(np.sqrt(y_uncer)), np.min(np.sqrt(y_uncer))])
        
        x1 = np.array(x1x2y_pred, dtype=object)[:,0].reshape(n_steps, n_steps)
        x2 = np.array(x1x2y_pred, dtype=object)[:,1].reshape(n_steps, n_steps)
            
        y_max2 = np.array(x1x2y_pred, dtype=object)[:,2].reshape(n_steps, n_steps)
        y_mean2 = np.array(x1x2y_pred, dtype=object)[:,3].reshape(n_steps, n_steps)
        y_min2 = np.array(x1x2y_pred, dtype=object)[:,4].reshape(n_steps, n_steps)
        
        y_uncer_max = np.array(x1x2y_uncer, dtype=object)[:,2].reshape(n_steps, n_steps)
        y_uncer_mean = np.array(x1x2y_uncer, dtype=object)[:,3].reshape(n_steps, n_steps)
        y_uncer_min = np.array(x1x2y_uncer, dtype=object)[:,4].reshape(n_steps, n_steps)

        fs = 20
        title_pad = 16
        
        fig,axes = plt.subplots(1, 3, figsize=(17, 4), sharey = False, sharex = False)
#         for ax, y in zip(axes,
#                            [y_max2, y_mean2, y_min2]):
#             c_plt1 = ax.contourf(x1, x2, y,cmap='plasma',extend='both')
        colorbar_offset = [40, 30, -10]
        for ax, c_offset, y in zip(axes,colorbar_offset,
                           [y_max2, y_mean2, y_min2]):
            c_plt1 = ax.contourf(x1, x2, y,levels = np.arange(21)+c_offset,cmap='plasma',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            # 修改 colorbar 刻度值显示（加 10）
            cbar_ticks = np.array([tick + 10 for tick in cbar.get_ticks()])
            cbar.set_ticks(cbar_ticks - 10)  # 将实际数据映射回原值
            cbar.set_ticklabels(cbar_ticks)  # 显示加 10 后的标签
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X2)[:, ind1], 
                       x_denormalizer(X2)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
            ax.scatter((X_new)[:, ind1], 
                       (X_new)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'green')
#             axes[0].contour(x1, x2, y_max2, colors='tan',levels=[150])
#             axes[0].contour(x1, x2, y_max2, colors='tan',levels=[0.4*(np.max(y_max2)-np.min(y_max2))+np.min(y_max2)])
            
            ax.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
            ax.set_ylabel(str(x_labels[ind2]),fontsize =  fs)
            
            x1_delta = (np.max(x1)-np.min(x1))*0.05
            x2_delta = (np.max(x2)-np.min(x2))*0.05
            ax.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
            ax.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
            
            ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
            if ind1==0:#R1
                    ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==1:#T
                ax.set_xticks([50, 60, 70, 80, 90, 100])
            if ind1==2:#R2
                ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==3:#H
                ax.set_xticks([20, 40, 60, 80])
            if ind1==4:#MACl%
                ax.set_xticks([0, 10, 20, 30, 40, 50])
            if ind1==5:#HT%
                ax.set_yticks([0.2, 0.8, 1.4, 2.0])
                
        axes[0].set_title('Delta_Peak max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('Delta_Peak mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('Delta_Peak min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()


# In[450]:


design = RandomDesign(parameter_space)
x_sampled = design.get_samples(200)
x_sampled = x_sampled
input_dim = 6
for i in range(input_dim):
    for j in range(input_dim-i-1):
        ind1 = i
        ind2 = j+i+1
        n_steps =21
        x1x2y_pred, x1x2y_uncer =[[],[]]
        for x1 in np.linspace(0, 1, n_steps):
            for x2 in np.linspace(0, 1, n_steps):
                x_temp = np.copy(x_sampled)
                x_temp[:,ind1] = x1
                x_temp[:,ind2] = x2
                y_pred = acquisition_constraint.evaluate(x_temp)
                y2 = y_pred
                x1_org = x_denormalizer(x_temp)[0,ind1]
                x2_org = x_denormalizer(x_temp)[0,ind2]
                x1x2y_pred.append([x1_org, x2_org, np.max(y2), np.mean(y2), np.min(y2)])
        
        x1 = np.array(x1x2y_pred, dtype=object)[:,0].reshape(n_steps, n_steps)
        x2 = np.array(x1x2y_pred, dtype=object)[:,1].reshape(n_steps, n_steps)
            
        y_max2 = np.array(x1x2y_pred, dtype=object)[:,2].reshape(n_steps, n_steps)
        y_mean2 = np.array(x1x2y_pred, dtype=object)[:,3].reshape(n_steps, n_steps)
        y_min2 = np.array(x1x2y_pred, dtype=object)[:,4].reshape(n_steps, n_steps)
   
        fs = 20
        title_pad = 16
        
        fig,axes = plt.subplots(1, 3, figsize=(17, 4), sharey = False, sharex = False)
#         for ax, y in zip(axes,
#                            [y_max2, y_mean2, y_min2]):
#             c_plt1 = ax.contourf(x1, x2, y,cmap='plasma',extend='both')
        colorbar_offset = [0.5, 0.1, 0]
        for ax, c_offset, y in zip(axes,colorbar_offset,
                           [y_max2, y_mean2, y_min2]):
            c_plt1 = ax.contourf(x1, x2, y/max(y_max2.flatten()),levels =np.arange(20)/2*0.05+c_offset,cmap='viridis',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X3)[:, ind1], 
                       x_denormalizer(X3)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
            ax.scatter((X_new)[:, ind1], 
                       (X_new)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'green')
            ax.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
            ax.set_ylabel(str(x_labels[ind2]),fontsize =  fs)
            
            x1_delta = (np.max(x1)-np.min(x1))*0.05
            x2_delta = (np.max(x2)-np.min(x2))*0.05
            ax.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
            ax.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
            
            ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
            if ind1==0:#R1
                    ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==1:#T
                ax.set_xticks([50, 60, 70, 80, 90, 100])
            if ind1==2:#R2
                ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==3:#H
                ax.set_xticks([20, 40, 60, 80])
            if ind1==4:#MACl%
                ax.set_xticks([0, 10, 20, 30, 40, 50])
            if ind1==5:#HT%
                ax.set_yticks([0.2, 0.8, 1.4, 2.0])
                
        axes[0].set_title('constraint fcn max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('constraint fcn mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('constraint fcn min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()


# In[452]:


design = RandomDesign(parameter_space)
x_sampled = design.get_samples(200)
x_sampled = x_sampled
input_dim = 6
for i in range(input_dim):
    for j in range(input_dim-i-1):
        ind1 = i
        ind2 = j+i+1
        n_steps =21
        x1x2y_pred, x1x2y_uncer =[[],[]]
        for x1 in np.linspace(0, 1, n_steps):
            for x2 in np.linspace(0, 1, n_steps):
                x_temp = np.copy(x_sampled)
                x_temp[:,ind1] = x1
                x_temp[:,ind2] = x2
                y_pred = acquisition1.evaluate(x_temp)
                y2 = y_pred
                x1_org = x_denormalizer(x_temp)[0,ind1]
                x2_org = x_denormalizer(x_temp)[0,ind2]
                x1x2y_pred.append([x1_org, x2_org, np.max(y2), np.mean(y2), np.min(y2)])
        
        x1 = np.array(x1x2y_pred, dtype=object)[:,0].reshape(n_steps, n_steps)
        x2 = np.array(x1x2y_pred, dtype=object)[:,1].reshape(n_steps, n_steps)
            
        y_max2 = np.array(x1x2y_pred, dtype=object)[:,2].reshape(n_steps, n_steps)
        y_mean2 = np.array(x1x2y_pred, dtype=object)[:,3].reshape(n_steps, n_steps)
        y_min2 = np.array(x1x2y_pred, dtype=object)[:,4].reshape(n_steps, n_steps)
   
        fs = 20
        title_pad = 16
        
        fig,axes = plt.subplots(1, 3, figsize=(17, 4), sharey = False, sharex = False)
#         for ax, y in zip(axes,
#                            [y_max2, y_mean2, y_min2]):
#             c_plt1 = ax.contourf(x1, x2, y/max(y_max2.flatten()),cmap='coolwarm',extend='both')
        colorbar_offset = [0.5, 0.1, 0]
        for ax, c_offset, y in zip(axes,colorbar_offset,
                           [y_max2, y_mean2, y_min2]):
            c_plt1 = ax.contourf(x1, x2, y/max(y_max2.flatten()),levels = np.arange(20)/2*0.05+c_offset,cmap='coolwarm',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X1)[:, ind1], 
                       x_denormalizer(X1)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
            ax.scatter((X_new)[:, ind1], 
                       (X_new)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'green')
#             axes[0].contour(x1, x2, y_max2, colors='darkslategrey',levels=[30])
            
            ax.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
            ax.set_ylabel(str(x_labels[ind2]),fontsize =  fs)
            
            x1_delta = (np.max(x1)-np.min(x1))*0.05
            x2_delta = (np.max(x2)-np.min(x2))*0.05
            ax.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
            ax.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
            
            ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
            if ind1==0:#R1
                    ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==1:#T
                ax.set_xticks([50, 60, 70, 80, 90, 100])
            if ind1==2:#R2
                ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==3:#H
                ax.set_xticks([20, 40, 60, 80])
            if ind1==4:#MACl%
                ax.set_xticks([0, 10, 20, 30, 40, 50])
            if ind1==5:#HT%
                ax.set_yticks([0.2, 0.8, 1.4, 2.0])
                
        axes[0].set_title('acqui fcn1 max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('acqui fcn1 mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('acqui fcn1 min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()


# In[454]:


design = RandomDesign(parameter_space)
x_sampled = design.get_samples(200)
x_sampled = x_sampled
input_dim = 6
for i in range(input_dim):
    for j in range(input_dim-i-1):
        ind1 = i
        ind2 = j+i+1
        n_steps =21
        x1x2y_pred, x1x2y_uncer =[[],[]]
        for x1 in np.linspace(0, 1, n_steps):
            for x2 in np.linspace(0, 1, n_steps):
                x_temp = np.copy(x_sampled)
                x_temp[:,ind1] = x1
                x_temp[:,ind2] = x2
                y_pred = acquisition2.evaluate(x_temp)
                y2 = y_pred
                x1_org = x_denormalizer(x_temp)[0,ind1]
                x2_org = x_denormalizer(x_temp)[0,ind2]
                x1x2y_pred.append([x1_org, x2_org, np.max(y2), np.mean(y2), np.min(y2)])
        
        x1 = np.array(x1x2y_pred, dtype=object)[:,0].reshape(n_steps, n_steps)
        x2 = np.array(x1x2y_pred, dtype=object)[:,1].reshape(n_steps, n_steps)
            
        y_max2 = np.array(x1x2y_pred, dtype=object)[:,2].reshape(n_steps, n_steps)
        y_mean2 = np.array(x1x2y_pred, dtype=object)[:,3].reshape(n_steps, n_steps)
        y_min2 = np.array(x1x2y_pred, dtype=object)[:,4].reshape(n_steps, n_steps)
   
        fs = 20
        title_pad = 16
        
        fig,axes = plt.subplots(1, 3, figsize=(17, 4), sharey = False, sharex = False)
#         for ax, y in zip(axes,
#                            [y_max2, y_mean2, y_min2]):
#             c_plt1 = ax.contourf(x1, x2, y/max(y_max2.flatten()),cmap='coolwarm',extend='both')
        colorbar_offset = [0.5, 0.1, 0]
        for ax, c_offset, y in zip(axes,colorbar_offset,
                           [y_max2, y_mean2, y_min2]):
            c_plt1 = ax.contourf(x1, x2, y/max(y_max2.flatten()),levels = np.arange(20)/2*0.05+c_offset,cmap='coolwarm',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X1)[:, ind1], 
                       x_denormalizer(X1)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
            ax.scatter((X_new)[:, ind1], 
                       (X_new)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'green')
#             axes[0].contour(x1, x2, y_max2, colors='darkslategrey',levels=[30])
            
            ax.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
            ax.set_ylabel(str(x_labels[ind2]),fontsize =  fs)
            
            x1_delta = (np.max(x1)-np.min(x1))*0.05
            x2_delta = (np.max(x2)-np.min(x2))*0.05
            ax.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
            ax.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
            
            ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
            if ind1==0:#R1
                    ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==1:#T
                ax.set_xticks([50, 60, 70, 80, 90, 100])
            if ind1==2:#R2
                ax.set_xticks([500, 1500, 2500, 3500, 4500])
            if ind1==3:#H
                ax.set_xticks([20, 40, 60, 80])
            if ind1==4:#MACl%
                ax.set_xticks([0, 10, 20, 30, 40, 50])
            if ind1==5:#HT%
                ax.set_yticks([0.2, 0.8, 1.4, 2.0])
                
        axes[0].set_title('acqui fcn2 max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('acqui fcn2 mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('acqui fcn2 min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




