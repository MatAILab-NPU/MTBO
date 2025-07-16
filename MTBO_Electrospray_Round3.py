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
import os
import scipy.stats
import math
from IPython.display import SVG, display
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.sans-serif'] = ["Arial"]
warnings.filterwarnings('ignore')


# In[2]:


MACl_min, MACl_max, MACl_step = [0, 30, 5] ## Unit: %, 7 steps
MACl_var = np.arange(MACl_min, MACl_max+MACl_step*0.1, MACl_step)
MACl_num = len(MACl_var)

DMF_min, DMF_max, DMF_step = [0, 90, 5] ## Unit: %, 19 steps
DMF_var = np.arange(DMF_min, DMF_max+DMF_step*0.1, DMF_step)
DMF_num = len(DMF_var)

V_min, V_max, V_step = [15, 30, 0.5] ## Unit: KV, 31 steps
V_var = np.arange(V_min, V_max+V_step*0.1, V_step)
V_num = len(V_var)

Q_min, Q_max, Q_step = [0.7, 1.7, 0.05] ## Unit: μL/min, # 21 steps
Q_var = np.arange(Q_min, Q_max+Q_step*0.1, Q_step)
Q_num = len(Q_var)

T_min, T_max, T_step = [100, 150, 5] ## Unit: ℃, # 11 steps
T_var = np.arange(T_min, T_max+T_step*0.1, T_step)
T_num = len(T_var)

t_min, t_max, t_step = [10, 30, 5] ## Unit: min, # 5 steps
t_var = np.arange(t_min, t_max+t_step*0.1, t_step)
t_num = len(t_var)


var_array = [MACl_var, DMF_var, 
             V_var, Q_var,
            T_var,t_var]

x_labels = ['MACl [%]', 
            'DMF [%]', 
            'V [KV]', 
            'Q [uL/min]',
            'T [degC]',
            't [min]']    


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
for MACl in MACl_var:
    for DMF in DMF_var:
        for V in V_var:
            for Q in Q_var:
                for T in T_var:
                    for t in t_var:
                        X_all_grid.append([MACl, DMF, V, Q, T,t])
X_all_grid = np.array(X_all_grid)
X_all_grid.shape


# In[6]:


x_eva=x_normalizer(X_all_grid)


# In[7]:


### Add/minus a half step to make sure the edge conditions have the same chance in nearest neighbors
parameter_space = ParameterSpace([ContinuousParameter('x1', 0-1/(MACl_num-1)/2, 1+1/(MACl_num-1)/2),
                                  ContinuousParameter('x2', 0-1/(DMF_num-1)/2,  1+1/(DMF_num-1)/2),
                                  ContinuousParameter('x3', 0-1/(V_num-1)/2,    1+1/(V_num-1)/2),
                                  ContinuousParameter('x4', 0-1/(Q_num-1)/2,    1+1/(Q_num-1)/2),
                                  ContinuousParameter('x5', 0-1/(T_num-1)/2,    1+1/(T_num-1)/2),
                                  ContinuousParameter('x6', 0-1/(t_num-1)/2,    1+1/(t_num-1)/2)])


# In[8]:


df_xrd = pd.read_csv('./data_3rd_cons.csv')
success_conditions = df_xrd[df_xrd['Success or Fail']==1]['ML Condition'].values
df_cons = df_xrd[df_xrd['ML Condition'].isin(success_conditions)]
df_cons


# In[9]:


df_film = pd.read_csv('./data_film_2.csv')
df_film


# In[10]:


df_exp1 = pd.read_csv('./data_3rd_obj1-paper.csv')
success_conditions = df_exp1[df_exp1['Success or Fail']==1]['ML Condition'].values
df_obj1 = df_exp1[df_exp1['ML Condition'].isin(success_conditions)]
df_obj1


# In[11]:


df_exp2 = pd.read_csv('./data_3rd_obj2.csv')
success_conditions = df_exp2[df_exp2['Success or Fail']==1]['ML Condition'].values
df_obj2 = df_exp2[df_exp2['ML Condition'].isin(success_conditions)]
df_obj2


# In[12]:


x1=df_cons.iloc[:,1:7].values#process conditions
x2=df_obj1.iloc[:,1:7].values
x3=df_obj2.iloc[:,1:7].values
x4=df_film.iloc[:,1:7].values
x1=x_normalizer(x1)
x2=x_normalizer(x2)
x3=x_normalizer(x3)
x4=x_normalizer(x4)
y1=np.transpose([df_cons.iloc[:,-2].values])#XRD(PbI:PVK)
y2=np.transpose([df_obj1.iloc[:,-2].values])#TRPL
y3=np.transpose([df_obj2.iloc[:,-2].values])#PL_std
y4=np.transpose([df_film.iloc[:,-1].values])#Film quality


# In[13]:


import GPy
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
X1, Y1 = [x1, y1]
X2, Y2 = [x2, y2]
X3, Y3 = [x3, y3]
X4, Y4 = [x4, y4]

input_dim = len(X1[0])
ker1 = GPy.kern.Matern52(input_dim = input_dim, ARD =True)
ker1.lengthscale.constrain_bounded(1e-2, 10)
ker1.variance.constrain_bounded(1e-2, 1e3)
yc_offset = np.mean(Y1)
# yc_offset = 0.1
model1_gpy = GPRegression(X1, Y1-yc_offset, ker1)#constraint acquisition computes the probability of < np.mean(XRD)
model1_gpy.Gaussian_noise.variance =0.1**2
model1_gpy.Gaussian_noise.variance.fix()
model1_gpy.randomize()
model1_gpy.optimize_restarts(num_restarts=20,verbose =False, messages=False)
objective_model1 = GPyModelWrapper(model1_gpy)
print(objective_model1.model.kern.lengthscale)
print(objective_model1.model.kern.variance)

input_dim = len(X2[0])
ker2 = GPy.kern.Matern52(input_dim = input_dim, ARD =True)
ker2.lengthscale.constrain_bounded(1e-2, 10)
ker2.variance.constrain_bounded(1e-2, 1e6)
model2_gpy = GPRegression(X2, -Y2, ker2)#Emukit is a minimization tool; need to make Y negative
model2_gpy.Gaussian_noise.variance = (0.1*np.mean(y2))**2
# model2_gpy.Gaussian_noise.variance = 10**2
model2_gpy.Gaussian_noise.variance.fix()
model2_gpy.randomize()
model2_gpy.optimize_restarts(num_restarts=20,verbose =False, messages=False)
objective_model2 = GPyModelWrapper(model2_gpy)
print(objective_model2.model.kern.lengthscale)
print(objective_model2.model.kern.variance)

input_dim = len(X3[0])
ker3 = GPy.kern.Matern52(input_dim = input_dim, ARD =True)
ker3.lengthscale.constrain_bounded(1e-2, 10)
ker3.variance.constrain_bounded(1e-2, 1e3)
model3_gpy = GPRegression(X3, Y3, ker3)
model3_gpy.Gaussian_noise.variance = 0.3**2
model3_gpy.Gaussian_noise.variance.fix()
model3_gpy.randomize()
model3_gpy.optimize_restarts(num_restarts=20,verbose =False, messages=False)
objective_model3 = GPyModelWrapper(model3_gpy)
print(objective_model3.model.kern.lengthscale)
print(objective_model3.model.kern.variance)

input_dim = len(X4[0])
ker4 = GPy.kern.Matern52(input_dim = input_dim, ARD =True)
ker4.lengthscale.constrain_bounded(1e-2, 1)
ker4.variance.constrain_bounded(1e-2, 1e3)
# ker4 += GPy.kern.White(input_dim = input_dim)
yc2_offset = 0.5
model4_gpy = GPRegression(X4, -(Y4-yc2_offset), ker4)
model4_gpy.Gaussian_noise.variance = 0.1**2
model4_gpy.Gaussian_noise.variance.fix()
model4_gpy.randomize()
model4_gpy.optimize_restarts(num_restarts=20,verbose =False, messages=False)
objective_model4 = GPyModelWrapper(model4_gpy)
print(objective_model4.model.kern.lengthscale)
print(objective_model4.model.kern.variance)

f_obj1 = objective_model1.model.predict
f_obj2 = objective_model2.model.predict
f_obj3 = objective_model3.model.predict
f_obj4 = objective_model4.model.predict

y1_pred, y1_uncer = f_obj1(X1)
y1_pred = y1_pred[:,-1]+yc_offset
y2_pred, y2_uncer = f_obj2(X2)
y2_pred = -y2_pred[:,-1]
y3_pred, y3_uncer = f_obj3(X3)
y3_pred = y3_pred[:,-1]
y4_pred, y4_uncer = f_obj4(X4)
y4_pred = -y4_pred[:,-1]+yc2_offset

y1_uncer = np.sqrt(y1_uncer[:,-1])
y2_uncer = np.sqrt(y2_uncer[:,-1])
y3_uncer = np.sqrt(y3_uncer[:,-1])
y4_uncer = np.sqrt(y4_uncer[:,-1])


# In[15]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
fig, axes = plt.subplots(1, 4, figsize=(5*5, 5))
fs = 18

lims1 = (-.1, 2.1)
axes[0].scatter(Y1[:,-1], y1_pred, alpha = 0.6, edgecolor = 'r', c = 'r')
axes[0].errorbar(Y1[:,-1], y1_pred, yerr = y1_uncer, ms = 0, 
             ls = '', capsize = 2, alpha = 0.6,
             color = 'gray', zorder = 0)
axes[0].plot(lims1, lims1, 'k--', alpha=0.75, zorder=0)
rmse_value = np.sqrt(mean_squared_error(Y1[:,-1], y1_pred))
title = 'GPR for XRD' + " (RMSE=%.2f" % rmse_value+' [%])'
axes[0].set_xlabel('Ground Truth', fontsize = fs)
axes[0].set_ylabel('Prediction', fontsize = fs)
axes[0].set_title(title, fontsize = fs)
mse = mean_squared_error
mse_xrd = mse(Y1[:,-1], y1_pred)
print ('xrd rmse: %.4f' % (np.sqrt(mse_xrd)))
rsquared_xrd = r2_score(Y1[:,-1], y1_pred)
print ('xrd R^2: %.4f' % (rsquared_xrd))
sprman_xrd = spearmanr(Y1[:,-1], y1_pred)
print ('xrd spearman: %.4f' % (sprman_xrd[0]))

lims2 = (-.1, 300.1)
axes[1].scatter(Y2[:,-1], y2_pred, alpha = 0.6, edgecolor = 'r', c = 'r')
axes[1].errorbar(Y2[:,-1], y2_pred, yerr = y2_uncer, ms = 0, 
             ls = '', capsize = 2, alpha = 0.6,
             color = 'gray', zorder = 0)
axes[1].plot(lims2, lims2, 'k--', alpha=0.75, zorder=0)
rmse_value = np.sqrt(mean_squared_error(Y2[:,-1], y2_pred))
title = 'GPR for TRPL' + " (RMSE=%.2f" % rmse_value+' [%])'
axes[1].set_xlabel('Ground Truth', fontsize = fs)
axes[1].set_ylabel('Prediction', fontsize = fs)
axes[1].set_title(title, fontsize = fs)
mse = mean_squared_error
mse_trpl = mse(Y2[:,-1], y2_pred)
print ('TRPL rmse: %.4f' % (np.sqrt(mse_trpl)))
rsquared_trpl = r2_score(Y2[:,-1], y2_pred)
print ('TRPL R^2: %.4f' % (rsquared_trpl))
sprman_trpl = spearmanr(Y2[:,-1], y2_pred)
print ('TRPL spearman: %.4f' % (sprman_trpl[0]))

lims3 = (-.1, 7.1)
axes[2].scatter(Y3[:,-1], y3_pred, alpha = 0.6, edgecolor = 'r', c = 'r')
axes[2].errorbar(Y3[:,-1], y3_pred, yerr = y3_uncer, ms = 0, 
             ls = '', capsize = 2, alpha = 0.6,
             color = 'gray', zorder = 0)
axes[2].plot(lims3, lims3, 'k--', alpha=0.75, zorder=0)
rmse_value = np.sqrt(mean_squared_error(Y3[:,-1], y3_pred))
title = 'GPR for PL' + " (RMSE=%.2f" % rmse_value+' [%])'
axes[2].set_xlabel('Ground Truth', fontsize = fs)
axes[2].set_ylabel('Prediction', fontsize = fs)
axes[2].set_title(title, fontsize = fs)
mse = mean_squared_error
mse_pl = mse(Y3[:,-1], y3_pred)
print ('PL rmse: %.4f' % (np.sqrt(mse_pl)))
rsquared_pl = r2_score(Y3[:,-1], y3_pred)
print ('PL R^2: %.4f' % (rsquared_pl))
sprman_pl = spearmanr(Y3[:,-1], y3_pred)
print ('PL spearman: %.4f' % (sprman_pl[0]))

lims4 = (-.1, 1.1)
axes[3].scatter(Y4[:,-1], y4_pred, alpha = 0.6, edgecolor = 'r', c = 'r')
axes[3].errorbar(Y4[:,-1], y4_pred, yerr = y4_uncer, ms = 0, 
             ls = '', capsize = 2, alpha = 0.6,
             color = 'gray', zorder = 0)
axes[3].plot(lims4, lims4, 'k--', alpha=0.75, zorder=0)
rmse_value = np.sqrt(mean_squared_error(Y4[:,-1], y4_pred))
title = 'GPR for Binary Film Quality'
# title = 'GPR for Film Quality' + " (RMSE=%.2f" % rmse_value+' [%])'
axes[3].set_xlabel('Ground Truth', fontsize = fs)
axes[3].set_ylabel('Prediction', fontsize = fs)
axes[3].set_title(title, fontsize = fs)
mse = mean_squared_error
mse_pl = mse(Y4[:,-1], y4_pred)
print ('Film rmse: %.4f' % (np.sqrt(mse_pl)))
rsquared_pl = r2_score(Y4[:,-1], y4_pred)
print ('Film R^2: %.4f' % (rsquared_pl))
sprman_pl = spearmanr(Y4[:,-1], y4_pred)
print ('Film spearman: %.4f' % (sprman_pl[0]))

plt.subplots_adjust(wspace = 0.4)


# In[61]:


from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound,ProbabilityOfFeasibility,ProbabilityOfImprovement
acquisition1 = NegativeLowerConfidenceBound(objective_model2,beta = 1)
acquisition2 = NegativeLowerConfidenceBound(objective_model3,beta = 1)
# acquisition_constraint1 = ScaledProbabilityOfFeasibility(objective_model1, max_value = 1, min_value = 0.5)
# acquisition_constraint2 = ScaledProbabilityOfFeasibility(objective_model4, max_value = 1, min_value = 0.5)
acquisition_constraint1 = ProbabilityOfFeasibility(objective_model1)
acquisition_constraint2 = ProbabilityOfFeasibility(objective_model4)
acquisition_constraint = acquisition_constraint1*acquisition_constraint2
obj1_plot = -acquisition1.evaluate(x_eva)#LCB of obj1
obj2_plot = -acquisition2.evaluate(x_eva)#LCB of obj2
cons_plot = acquisition_constraint.evaluate(x_eva)

tolerances = np.array([-99,1.97]) 
absolutes = [True, True]
goals = ['min', 'min'] 
chimera = Chimera(tolerances=tolerances, absolutes=absolutes, goals=goals)
obj = np.array([obj1_plot.T[0], obj2_plot.T[0]])
scalarized = chimera.scalarize(obj.T)
cons_acq = (1-scalarized)*cons_plot.T[0]#constrained acquisition


# In[82]:


np.random.seed(10) # to make sure the random results is reproducible 
bs=10
sort_index = np.argsort(cons_acq, axis =0)
X_new = []
top = 4760 # top .01% = 476/4762065 You should set it according to your confidence in the model (balance of exploitation and exploration)
for i in sort_index[-top:]:
        X_new.append(X_all_grid[i])
X_new=np.array(X_new)
bs_index = [np.random.randint(top) for i in np.arange(bs)]
X_new = X_new[bs_index]
# acq_new1 = acquisition1.evaluate(x_normalizer(X_new))
# acq_new2 = acquisition2.evaluate(x_normalizer(X_new))
cons_pr = acquisition_constraint.evaluate(x_normalizer(X_new))
cons_pr = cons_pr/np.max(cons_plot)
idx=sort_index[-top:][bs_index]
final_acq=cons_acq[idx]
final_acq=final_acq/np.max(cons_acq)
y_new_pred1, y_new_uncer1 = f_obj2(x_normalizer(X_new))
y_new_pred2, y_new_uncer2 = f_obj3(x_normalizer(X_new))
df_Xnew = pd.DataFrame(X_new, columns = x_labels)
df_all = pd.concat([df_cons.iloc[:,1:7], df_Xnew])
df_all_ = df_all.drop_duplicates()
df_Xnew = df_all_.iloc[len(df_cons):len(df_cons)+bs]
df_Xnew = df_Xnew.sort_values(by=list(df_cons.columns[1:7]), ignore_index = True)
df_Xnew.index = np.arange(len(df_Xnew))+len(df_xrd)
print('New X:',len(df_Xnew))
df_Xnew


# In[83]:


df_x = df_Xnew
df_cols = df_cons.columns
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


# In[19]:


fig, axes = plt.subplots(4, 1, figsize=(8, 22), sharey = False, sharex = False)
fs = 20
film_xrd = df_xrd.sort_values('ML Condition').iloc[:,-2].values
exp_xrd = film_xrd.reshape(-1,1)

data_xrd = df_cons.sort_values('ML Condition').iloc[:,[0,-2]].values
exp_cond = data_xrd[:,0]

f_obj = objective_model1.model.predict
y_pred, y_uncer = f_obj(X1)
y_pred = y_pred[:,-1]+yc_offset
y_uncer = np.sqrt(y_uncer[:,-1])

unsuccess_idx=df_xrd[df_xrd['Success or Fail']==0]['ML Condition'].values
unsuccess_film=df_xrd[df_xrd['Success or Fail']==0]['Success or Fail'].values
unsuccess_film=unsuccess_film-0.25
all_cond = np.concatenate([data_xrd[:,0], np.transpose(unsuccess_idx)])
all_cond = all_cond[np.argsort(all_cond)]

axes[0].scatter(unsuccess_idx, unsuccess_film,
                facecolor = 'black',
                edgecolor = 'black',
                s = 20, label = 'failed')

axes[0].scatter(exp_cond, y1, #facecolor = 'none',
            edgecolor = 'navy', s = 20, alpha = 0.6, label = 'experiment')

# axes[0].plot(exp_cond, np.minimum.accumulate(y1), 
#          marker = 'o', ms = 0, c = 'black')

axes[0].scatter(exp_cond, y_pred,
                s = 50, facecolors='none', alpha = 0.6, edgecolor = 'gray', label = 'predicted')
axes[0].errorbar(exp_cond, y_pred, yerr = y_uncer,  
                 ms = 1, ls = '', capsize = 2, alpha = 0.6, 
                 color = 'gray', zorder = 0)


y_pred_new, y_uncer_new = f_obj(x_normalizer(df_Xnew.values))
y_pred_new = y_pred_new[:,-1]+yc_offset
y_uncer_new = np.sqrt(y_uncer_new[:,-1])

axes[0].scatter(np.arange(len(df_Xnew))+len(exp_xrd), y_pred_new,
                s = 50, facecolors='none', alpha = 0.6, edgecolor = 'darkgreen', label = 'suggested')
axes[0].errorbar(np.arange(len(df_Xnew))+len(exp_xrd), y_pred_new, yerr = y_uncer_new,  
                 ms = 0, ls = '', capsize = 2, alpha = 0.6, 
                 color = 'darkgreen', zorder = 0)


axes[0].set_ylabel('Current Best XRD', fontsize = 20)
axes[0].set_xlabel('Process Condition', fontsize = 20)

axes[0].set_ylim(-.5, 2.1)
axes[0].set_xlim(-1, 45)
axes[0].set_xticks(np.arange(0,45,5))
axes[0].legend(fontsize = fs*0.7)

film_life = df_exp1.sort_values('ML Condition').iloc[:,-2].values
exp_life = film_life.reshape(-1,1)

data_life = df_obj1.sort_values('ML Condition').iloc[:,[0,-2]].values
exp_cond = data_life[:,0]

f_obj = objective_model2.model.predict
y_pred, y_uncer = f_obj(X2)
y_pred = -y_pred[:,-1]
y_uncer = np.sqrt(y_uncer[:,-1])

unsuccess_idx=df_exp1[df_exp1['Success or Fail']==0]['ML Condition'].values
unsuccess_film=df_exp1[df_exp1['Success or Fail']==0]['Success or Fail'].values
unsuccess_film=unsuccess_film
all_cond = np.concatenate([data_life[:,0], np.transpose(unsuccess_idx)])
all_cond = all_cond[np.argsort(all_cond)]

axes[1].scatter(unsuccess_idx, unsuccess_film,
                facecolor = 'black',
                edgecolor = 'black',
                s = 20, label = 'failed')

axes[1].scatter(exp_cond, y2, #facecolor = 'none',
            edgecolor = 'navy', s = 20, alpha = 0.6, label = 'experiment')

# axes[1].plot(exp_cond, np.maximum.accumulate(y2), 
#          marker = 'o', ms = 0, c = 'black')

axes[1].scatter(exp_cond, y_pred,
                s = 50, facecolors='none', alpha = 0.6, edgecolor = 'gray', label = 'predicted')
axes[1].errorbar(exp_cond, y_pred, yerr = y_uncer,  
                 ms = 1, ls = '', capsize = 2, alpha = 0.6, 
                 color = 'gray', zorder = 0)


y_pred_new, y_uncer_new = f_obj(x_normalizer(df_Xnew.values))
y_pred_new = -y_pred_new[:,-1]
y_uncer_new = np.sqrt(y_uncer_new[:,-1])

axes[1].scatter(np.arange(len(df_Xnew))+len(exp_life), y_pred_new,
                s = 50, facecolors='none', alpha = 0.6, edgecolor = 'darkgreen', label = 'suggested')
axes[1].errorbar(np.arange(len(df_Xnew))+len(exp_life), y_pred_new, yerr = y_uncer_new,  
                 ms = 0, ls = '', capsize = 2, alpha = 0.6, 
                 color = 'darkgreen', zorder = 0)

axes[1].set_ylabel('Current Best TRPL', fontsize = 20)
axes[1].set_xlabel('Process Condition', fontsize = 20)

axes[1].set_ylim(-50, 350)
axes[1].set_xlim(-1, 45)
axes[1].set_xticks(np.arange(0,45,5))
axes[1].legend(fontsize = fs*0.7)


film_uniform = df_exp2.sort_values('ML Condition').iloc[:,-2].values
exp_uniform = film_uniform.reshape(-1,1)

data_uniform = df_obj2.sort_values('ML Condition').iloc[:,[0,-2]].values
exp_cond = data_uniform[:,0]

f_obj = objective_model3.model.predict
y_pred, y_uncer = f_obj(X3)
y_pred = y_pred[:,-1]
y_uncer = np.sqrt(y_uncer[:,-1])

unsuccess_idx=df_exp2[df_exp2['Success or Fail']==0]['ML Condition'].values
unsuccess_film=df_exp2[df_exp2['Success or Fail']==0]['Success or Fail'].values
unsuccess_film=unsuccess_film
all_cond = np.concatenate([data_uniform[:,0], np.transpose(unsuccess_idx)])
all_cond = all_cond[np.argsort(all_cond)]

axes[2].scatter(unsuccess_idx, unsuccess_film,
                facecolor = 'black',
                edgecolor = 'black',
                s = 20, label = 'failed')

axes[2].scatter(exp_cond, y3, #facecolor = 'none',
            edgecolor = 'navy', s = 20, alpha = 0.6, label = 'experiment')

# axes[2].plot(exp_cond, np.minimum.accumulate(y3), 
#          marker = 'o', ms = 0, c = 'black')

axes[2].scatter(exp_cond, y_pred,
                s = 50, facecolors='none', alpha = 0.6, edgecolor = 'gray', label = 'predicted')
axes[2].errorbar(exp_cond, y_pred, yerr = y_uncer,  
                 ms = 1, ls = '', capsize = 2, alpha = 0.6, 
                 color = 'gray', zorder = 0)


y_pred_new, y_uncer_new = f_obj(x_normalizer(df_Xnew.values))
y_pred_new = y_pred_new[:,-1]
y_uncer_new = np.sqrt(y_uncer_new[:,-1])

axes[2].scatter(np.arange(len(df_Xnew))+len(exp_uniform), y_pred_new,
                s = 50, facecolors='none', alpha = 0.6, edgecolor = 'darkgreen', label = 'suggested')
axes[2].errorbar(np.arange(len(df_Xnew))+len(exp_uniform), y_pred_new, yerr = y_uncer_new,  
                 ms = 0, ls = '', capsize = 2, alpha = 0.6, 
                 color = 'darkgreen', zorder = 0)


axes[2].set_ylabel('Current Best PL', fontsize = 20)
axes[2].set_xlabel('Process Condition', fontsize = 20)

axes[2].set_ylim(-1, 7)
axes[2].set_xlim(-1, 45)
axes[2].set_xticks(np.arange(0,45,5))
axes[2].legend(fontsize = fs*0.7,loc=1)


film_quality = df_xrd.sort_values('ML Condition').iloc[:,-3].values
exp_quality = film_quality.reshape(-1,1)

data_quality = df_cons.sort_values('ML Condition').iloc[:,[0,-3]].values
exp_cond = data_quality[:,0]

f_obj = objective_model4.model.predict
y_pred, y_uncer = f_obj(X4)
y_pred = -y_pred[:,-1]+yc2_offset
y_uncer = np.sqrt(y_uncer[:,-1])

unsuccess_idx=df_xrd[df_xrd['Film Quality']==0]['ML Condition'].values
unsuccess_film=df_xrd[df_xrd['Film Quality']==0]['Film Quality'].values

success_idx=df_xrd[df_xrd['Film Quality']==1]['ML Condition'].values
success_film=df_xrd[df_xrd['Film Quality']==1]['Film Quality'].values
# all_cond = np.concatenate([data_quality[:,0], np.transpose(unsuccess_idx)])
# all_cond = all_cond[np.argsort(all_cond)]

axes[3].scatter(unsuccess_idx, unsuccess_film,
                facecolor = 'black',
                edgecolor = 'black',
                s = 20, label = 'failed')

axes[3].scatter(success_idx, success_film, #facecolor = 'none',
            edgecolor = 'navy', s = 20, alpha = 0.6, label = 'experiment')


y_pred_new, y_uncer_new = f_obj(x_normalizer(df_Xnew.values))
y_pred_new = -y_pred_new[:,-1]+yc2_offset
y_uncer_new = np.sqrt(y_uncer_new[:,-1])

axes[3].scatter(np.arange(len(df_Xnew))+len(exp_quality), y_pred_new,
                s = 50, facecolors='none', alpha = 0.6, edgecolor = 'darkgreen', label = 'suggested')
axes[3].errorbar(np.arange(len(df_Xnew))+len(exp_quality), y_pred_new, yerr = y_uncer_new,  
                 ms = 0, ls = '', capsize = 2, alpha = 0.6, 
                 color = 'darkgreen', zorder = 0)
# axes[3].hlines([yc_offset], -5,45, linestyles="--", colors='red',linewidth=2)


axes[3].set_ylabel('Current Best Film', fontsize = 20)
axes[3].set_xlabel('Process Condition', fontsize = 20)

axes[3].set_ylim(-.1, 1.3)
axes[3].set_xlim(-1, 45)
axes[3].set_xticks(np.arange(0,45,5))
axes[3].legend(fontsize = fs*0.7)

for ax in axes:
    ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8, grid_alpha = 0.5)
    ax.grid(True, linestyle='-.')
plt.subplots_adjust(wspace = 0.8)

plt.show()


# In[87]:


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey = False)
fs = 20

# axes[0].plot(np.arange(len(df_Xnew))+1+len(X2), acq_new1, marker = 'o',
#                 ms = 2, alpha = 0.6, color = 'orange', label = 'raw acqui1')
# axes[0].plot(np.arange(len(df_Xnew))+1+len(X2), acq_new1, marker = 'o',
#                 ms = 2, alpha = 0.6, color = 'navy', label = 'raw acqui2')
axes[0].plot(np.arange(len(df_Xnew))+len(exp_xrd), cons_pr, marker = 'o',
                ms = 2, alpha = 0.6, color = 'red', label = 'constr prob')
axes[0].plot(np.arange(len(df_Xnew))+len(exp_xrd), final_acq, marker = 'o',
                ms = 2, alpha = 0.6, color = 'darkgreen', label = 'final acqui')

axes[0].set_ylim(0., 1)
axes[0].set_xlim(-1, 50)
axes[0].set_xticks(np.arange(0,50,10))
axes[0].set_ylabel('Acquisition Probability', fontsize = fs)
axes[0].set_xlabel('Process Condition', fontsize = fs)
axes[0].legend(fontsize = fs*0.7,loc=4)

axes[1].axis("off")
for ax in axes:
    ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8, grid_alpha = 0.5)
    ax.grid(True, linestyle='-.')
plt.subplots_adjust(wspace = 0.4)

plt.show()


# In[116]:


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
                y2 = y_pred+yc_offset
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
#             c_plt1 = ax.contourf(x1, x2, y,cmap='plasma',extend='both')
        colorbar_offset = [1.2, 0.5, 0]
        for ax, c_offset, y in zip(axes,colorbar_offset,
                           [y_max2, y_mean2, y_min2]):
            c_plt1 = ax.contourf(x1, x2, y,levels = np.arange(10)*0.05+c_offset,cmap='viridis',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X1)[:, ind1], 
                       x_denormalizer(X1)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
            ax.scatter((X_new)[:, ind1], 
                       (X_new)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'green')
#             axes[2].contour(x1, x2, y_min2, colors='r',levels=[yc_offset])
            
            ax.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
            ax.set_ylabel(str(x_labels[ind2]),fontsize =  fs)
            
            x1_delta = (np.max(x1)-np.min(x1))*0.05
            x2_delta = (np.max(x2)-np.min(x2))*0.05
            ax.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
            ax.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
            
            ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
            if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
            if ind1==1:#DMF%
                ax.set_xticks([0, 30, 60, 90])
            if ind1==2:#V
                ax.set_xticks([15, 20, 25, 30])
            if ind1==3:#Q
                ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
            if ind1==4:#T
                ax.set_xticks([100, 110, 120, 130, 140, 150])
            if ind1==5:#t
                ax.set_yticks([10, 15, 20, 25, 30])
                
        axes[0].set_title('xrd max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('xrd mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('xrd min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()
        
        fig,axes = plt.subplots(1, 1, figsize=(4.5, 3.5), sharey =False, sharex = False) 
#         c_plt3 = axes.contourf(x1, x2, y_uncer_max, cmap='plasma',extend='both')
        colorbar_offset = [0.5]
        c_plt3 = axes.contourf(x1, x2, y_uncer_max, levels = np.arange(11)*0.01+colorbar_offset, cmap='viridis',extend='both')
        cbar = fig.colorbar(c_plt3, ax = axes)
        axes.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
        axes.set_ylabel(str(x_labels[ind2]),fontsize =  fs)

        x1_delta = (np.max(x1)-np.min(x1))*0.05
        x2_delta = (np.max(x2)-np.min(x2))*0.05
        axes.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
        axes.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
        axes.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
        if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
        if ind1==1:#DMF%
            ax.set_xticks([0, 30, 60, 90])
        if ind1==2:#V
            ax.set_xticks([15, 20, 25, 30])
        if ind1==3:#Q
            ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
        if ind1==4:#T
            ax.set_xticks([100, 110, 120, 130, 140, 150])
        if ind1==5:#t
            ax.set_yticks([10, 15, 20, 25, 30])
            
#         axes.scatter(x_denormalizer(X1)[:, 0], 
#                        x_denormalizer(X1)[:, 1], 
#                        s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
        axes.set_title('variance max', pad = title_pad,fontsize =  fs)


# In[117]:


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
                y_pred, y_uncer = f_obj4(x_temp)
                y2 = -y_pred+yc2_offset
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
            c_plt1 = ax.contourf(x1, x2, y,levels = np.arange(11)*0.05+c_offset,cmap='viridis',extend = 'both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X4)[:, ind1], 
                       x_denormalizer(X4)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
            ax.scatter((X_new)[:, ind1], 
                       (X_new)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'green')
            axes[0].contour(x1, x2, y_min2, colors='r',levels=[yc2_offset])
            
            ax.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
            ax.set_ylabel(str(x_labels[ind2]),fontsize =  fs)
            
            x1_delta = (np.max(x1)-np.min(x1))*0.05
            x2_delta = (np.max(x2)-np.min(x2))*0.05
            ax.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
            ax.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
            
            ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
            if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
            if ind1==1:#DMF%
                ax.set_xticks([0, 30, 60, 90])
            if ind1==2:#V
                ax.set_xticks([15, 20, 25, 30])
            if ind1==3:#Q
                ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
            if ind1==4:#T
                ax.set_xticks([100, 110, 120, 130, 140, 150])
            if ind1==5:#t
                ax.set_yticks([10, 15, 20, 25, 30])
                
        axes[0].set_title('film quality max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('film quality mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('film quality min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()
        
        fig,axes = plt.subplots(1, 1, figsize=(4.5, 3.5), sharey =False, sharex = False) 
#         c_plt3 = axes.contourf(x1, x2, y_uncer_max, cmap='plasma',extend='both')
        colorbar_offset = [0.45]
        c_plt3 = axes.contourf(x1, x2, y_uncer_max, levels = np.arange(11)*0.01+colorbar_offset, cmap='viridis',extend='both')
        cbar = fig.colorbar(c_plt3, ax = axes)
        axes.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
        axes.set_ylabel(str(x_labels[ind2]),fontsize =  fs)

        x1_delta = (np.max(x1)-np.min(x1))*0.05
        x2_delta = (np.max(x2)-np.min(x2))*0.05
        axes.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
        axes.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
        axes.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
        if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
        if ind1==1:#DMF%
            ax.set_xticks([0, 30, 60, 90])
        if ind1==2:#V
            ax.set_xticks([15, 20, 25, 30])
        if ind1==3:#Q
            ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
        if ind1==4:#T
            ax.set_xticks([100, 110, 120, 130, 140, 150])
        if ind1==5:#t
            ax.set_yticks([10, 15, 20, 25, 30])
            
#         axes.scatter(x_denormalizer(X1)[:, 0], 
#                        x_denormalizer(X1)[:, 1], 
#                        s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
        axes.set_title('variance max', pad = title_pad,fontsize =  fs)


# In[118]:


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
                y2 = -y_pred
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
        colorbar_offset = [75, 25, 0]
        for ax, c_offset, y in zip(axes,colorbar_offset,
                           [y_max2, y_mean2, y_min2]):
            c_plt1 = ax.contourf(x1, x2, y,levels = np.arange(16)*25+c_offset,cmap='plasma',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X2)[:, ind1], 
                       x_denormalizer(X2)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
            ax.scatter((X_new)[:, ind1], 
                       (X_new)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'green')
#             axes[0].contour(x1, x2, y_max2, colors='b',levels=[300])
            
            ax.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
            ax.set_ylabel(str(x_labels[ind2]),fontsize =  fs)
            
            x1_delta = (np.max(x1)-np.min(x1))*0.05
            x2_delta = (np.max(x2)-np.min(x2))*0.05
            ax.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
            ax.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
            
            ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
            if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
            if ind1==1:#DMF%
                ax.set_xticks([0, 30, 60, 90])
            if ind1==2:#V
                ax.set_xticks([15, 20, 25, 30])
            if ind1==3:#Q
                ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
            if ind1==4:#T
                ax.set_xticks([100, 110, 120, 130, 140, 150])
            if ind1==5:#t
                ax.set_yticks([10, 15, 20, 25, 30])
                
        axes[0].set_title('TRPL max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('TRPL mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('TRPL min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()
        
        fig,axes = plt.subplots(1, 1, figsize=(4.5, 3.5), sharey =False, sharex = False) 
#         c_plt3 = axes.contourf(x1, x2, y_uncer_max, cmap='plasma',extend='both')
        colorbar_offset = [130]
        c_plt3 = axes.contourf(x1, x2, y_uncer_max, levels = np.arange(11)+colorbar_offset, cmap='plasma',extend='both')
        cbar = fig.colorbar(c_plt3, ax = axes)
        axes.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
        axes.set_ylabel(str(x_labels[ind2]),fontsize =  fs)

        x1_delta = (np.max(x1)-np.min(x1))*0.05
        x2_delta = (np.max(x2)-np.min(x2))*0.05
        axes.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
        axes.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
        axes.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
        if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
        if ind1==1:#DMF%
            ax.set_xticks([0, 30, 60, 90])
        if ind1==2:#V
            ax.set_xticks([15, 20, 25, 30])
        if ind1==3:#Q
            ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
        if ind1==4:#T
            ax.set_xticks([100, 110, 120, 130, 140, 150])
        if ind1==5:#t
            ax.set_yticks([10, 15, 20, 25, 30])
            
#         axes.scatter(x_denormalizer(X1)[:, 0], 
#                        x_denormalizer(X1)[:, 1], 
#                        s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
        axes.set_title('variance max', pad = title_pad,fontsize =  fs)


# In[90]:


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
        colorbar_offset = [3, 2, 0]
        for ax, c_offset, y in zip(axes,colorbar_offset,
                           [y_max2, y_mean2, y_min2]):
            c_plt1 = ax.contourf(x1, x2, y,levels = np.arange(21)/10+c_offset,cmap='plasma',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X3)[:, ind1], 
                       x_denormalizer(X3)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
            ax.scatter((X_new)[:, ind1], 
                       (X_new)[:, ind2], 
                       s = 50, facecolors='none', alpha = 0.9, edgecolor = 'green')
#             axes[2].contour(x1, x2, y_min2, colors='b',levels=[1])
            
            ax.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
            ax.set_ylabel(str(x_labels[ind2]),fontsize =  fs)
            
            x1_delta = (np.max(x1)-np.min(x1))*0.05
            x2_delta = (np.max(x2)-np.min(x2))*0.05
            ax.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
            ax.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
            
            ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
            if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
            if ind1==1:#DMF%
                ax.set_xticks([0, 30, 60, 90])
            if ind1==2:#V
                ax.set_xticks([15, 20, 25, 30])
            if ind1==3:#Q
                ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
            if ind1==4:#T
                ax.set_xticks([100, 110, 120, 130, 140, 150])
            if ind1==5:#t
                ax.set_yticks([10, 15, 20, 25, 30])
                
        axes[0].set_title('PL max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('PL mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('PL min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()
        
        fig,axes = plt.subplots(1, 1, figsize=(4.5, 3.5), sharey =False, sharex = False) 
#         c_plt3 = axes.contourf(x1, x2, y_uncer_max, cmap='plasma',extend='both')
        colorbar_offset = [1]
        c_plt3 = axes.contourf(x1, x2, y_uncer_max, levels = np.arange(21)*0.1+colorbar_offset, cmap='plasma',extend='both')
        cbar = fig.colorbar(c_plt3, ax = axes)
        axes.set_xlabel(str(x_labels[ind1]),fontsize =  fs)
        axes.set_ylabel(str(x_labels[ind2]),fontsize =  fs)

        x1_delta = (np.max(x1)-np.min(x1))*0.05
        x2_delta = (np.max(x2)-np.min(x2))*0.05
        axes.set_xlim(np.min(x1)-x1_delta, np.max(x1)+x1_delta)
        axes.set_ylim(np.min(x2)-x2_delta, np.max(x2)+x2_delta)
        axes.tick_params(direction='in', length=5, width=1, labelsize = fs*.8)#, grid_alpha = 0.5
        if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
        if ind1==1:#DMF%
            ax.set_xticks([0, 30, 60, 90])
        if ind1==2:#V
            ax.set_xticks([15, 20, 25, 30])
        if ind1==3:#Q
            ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
        if ind1==4:#T
            ax.set_xticks([100, 110, 120, 130, 140, 150])
        if ind1==5:#t
            ax.set_yticks([10, 15, 20, 25, 30])
            
#         axes.scatter(x_denormalizer(X1)[:, 0], 
#                        x_denormalizer(X1)[:, 1], 
#                        s = 50, facecolors='none', alpha = 0.9, edgecolor = 'red')
        axes.set_title('variance max', pad = title_pad,fontsize =  fs)


# In[120]:


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
                y_pred = acquisition_constraint1.evaluate(x_temp)
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
            c_plt1 = ax.contourf(x1, x2, y/max(y_max2.flatten()),levels =np.arange(10)*0.05+c_offset,cmap='viridis',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X1)[:, ind1], 
                       x_denormalizer(X1)[:, ind2], 
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
            if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
            if ind1==1:#DMF%
                ax.set_xticks([0, 30, 60, 90])
            if ind1==2:#V
                ax.set_xticks([15, 20, 25, 30])
            if ind1==3:#Q
                ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
            if ind1==4:#T
                ax.set_xticks([100, 110, 120, 130, 140, 150])
            if ind1==5:#t
                ax.set_yticks([10, 15, 20, 25, 30])
                
        axes[0].set_title('constraint fcn1 max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('constraint fcn1 mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('constraint fcn1 min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()


# In[121]:


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
                y_pred = acquisition_constraint2.evaluate(x_temp)
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
            c_plt1 = ax.contourf(x1, x2, y/max(y_max2.flatten()),levels =np.arange(10)*0.05+c_offset,cmap='plasma',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X4)[:, ind1], 
                       x_denormalizer(X4)[:, ind2], 
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
            if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
            if ind1==1:#DMF%
                ax.set_xticks([0, 30, 60, 90])
            if ind1==2:#V
                ax.set_xticks([15, 20, 25, 30])
            if ind1==3:#Q
                ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
            if ind1==4:#T
                ax.set_xticks([100, 110, 120, 130, 140, 150])
            if ind1==5:#t
                ax.set_yticks([10, 15, 20, 25, 30])
                
        axes[0].set_title('constraint fcn2 max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('constraint fcn2 mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('constraint fcn2 min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()


# In[122]:


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
            c_plt1 = ax.contourf(x1, x2, y/max(y_max2.flatten()),levels = np.arange(10)*0.05+c_offset,cmap='coolwarm',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X2)[:, ind1], 
                       x_denormalizer(X2)[:, ind2], 
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
            if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
            if ind1==1:#DMF%
                ax.set_xticks([0, 30, 60, 90])
            if ind1==2:#V
                ax.set_xticks([15, 20, 25, 30])
            if ind1==3:#Q
                ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
            if ind1==4:#T
                ax.set_xticks([100, 110, 120, 130, 140, 150])
            if ind1==5:#t
                ax.set_yticks([10, 15, 20, 25, 30])
                
        axes[0].set_title('acqui fcn1 max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('acqui fcn1 mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('acqui fcn1 min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()


# In[123]:


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
            c_plt1 = ax.contourf(x1, x2, y/max(y_max2.flatten()),levels =np.arange(10)*0.05+c_offset,cmap='coolwarm',extend='both')
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
            if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
            if ind1==1:#DMF%
                ax.set_xticks([0, 30, 60, 90])
            if ind1==2:#V
                ax.set_xticks([15, 20, 25, 30])
            if ind1==3:#Q
                ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
            if ind1==4:#T
                ax.set_xticks([100, 110, 120, 130, 140, 150])
            if ind1==5:#t
                ax.set_yticks([10, 15, 20, 25, 30])
                
        axes[0].set_title('acqui fcn2 max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('acqui fcn2 mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('acqui fcn2 min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()


# In[124]:


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
#             c_plt1 = ax.contourf(x1, x2, y/max(y_max2.flatten()),cmap='coolwarm',extend='both')
        colorbar_offset = [0.5, 0.1, 0]
        for ax, c_offset, y in zip(axes,colorbar_offset,
                           [y_max2, y_mean2, y_min2]):
            c_plt1 = ax.contourf(x1, x2, y/max(y_max2.flatten()),levels =np.arange(10)*0.05+c_offset,cmap='coolwarm',extend='both')
#             c_plt1 = ax.contourf(x1, x2, y,levels =np.arange(10)*0.05+c_offset,cmap='coolwarm',extend='both')
            cbar = fig.colorbar(c_plt1, ax= ax)
            cbar.ax.tick_params(labelsize=fs*0.8)
            ax.scatter(x_denormalizer(X4)[:, ind1], 
                       x_denormalizer(X4)[:, ind2], 
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
            if ind1==0:#MACl%
                ax.set_xticks([0, 10, 20, 30])
            if ind1==1:#DMF%
                ax.set_xticks([0, 30, 60, 90])
            if ind1==2:#V
                ax.set_xticks([15, 20, 25, 30])
            if ind1==3:#Q
                ax.set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
            if ind1==4:#T
                ax.set_xticks([100, 110, 120, 130, 140, 150])
            if ind1==5:#t
                ax.set_yticks([10, 15, 20, 25, 30])
                
        axes[0].set_title('constr prob fcn max', pad = title_pad,fontsize =  fs)
        axes[1].set_title('constr prob fcn mean', pad = title_pad,fontsize =  fs)
        axes[2].set_title('constr prob fcn min', pad = title_pad,fontsize =  fs)

        plt.subplots_adjust(wspace = 0.3)
        plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




