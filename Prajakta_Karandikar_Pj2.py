#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import scipy
import matplotlib.pyplot as plt


# # Q1 part a

# In[4]:


#Generate the sample list
NumofSamples = 1000
a= -3
b= 2
Sample =np.array(np.random.uniform(a,b,NumofSamples))


# In[5]:


n, bins, patches = plt.hist(Sample, bins=10, color='red',
                            alpha=0.5, rwidth = 0.9)
plt.grid(axis='y', alpha=0.5)
plt.xlabel('Sample Value')
plt.ylabel('Frequency')
plt.title('Histogram')


# # Q1 part b

# In[6]:


#find the mean and variance from samples as well as theoretical
Sample_mean = np.mean(Sample)
Mean_th = (b+a)/2
Sample_var = np.var(Sample)
Var_th = ((b-a)**2)/12
print('Mean of the Sample = {}'.format(Sample_mean))
print('Theoretical Mean = {}'.format(Mean_th))
print('Variance of the Samples = {}'.format(Sample_var))
print('Theoretical Variance = {}'.format(Var_th))


# # Q1 part c

# In[10]:


import bootstrapped.bootstrap as bts
import bootstrapped.stats_functions as bts_stat

mean2 = bts.bootstrap(Sample, stat_func = bts_stat.mean)
print(mean2)
var2 = bts.bootstrap(Sample, stat_func = bts_stat.std)
print(var2)


# # Q2 part a

# In[11]:


# define the number of samples and generate a uniform random vector
N = 10000
X = np.array(np.random.rand(1,N))
#Compute covariance
k=1
X_k = np.array(np.roll(X,1))
covar = np.cov(X,X_k)
print('The Covariance of X_k,X_k+1 is:')
print(covar)


# # Q2 part b

# In[12]:


X_k_1 =np.array(np.roll(X,-1))
X_k_2 =np.array(np.roll(X,-2))
X_k_3 =np.array(np.roll(X,-3))
Y =(X -(2*X_k_1) + ((X_k_2)/2) -(X_k_3))
print('The covariance of COV[X_k, Y_k] is ')
Covar = np.cov(X,Y)
print(Covar)


# # Q3 part a

# In[13]:


M= 10
num = 1000
samples = np.random.randint(0,M,num)
n, bins, patches = plt.hist(samples, bins=10, color='red',
                            alpha=0.5, rwidth = 0.9)
plt.grid(axis='y', alpha=0.5)
plt.xlabel('Samples')
plt.ylabel('Output')
plt.title('Histogram')


# # Q3 part b

# In[19]:


#find observed and expected values
print('Observed values:')
print(n)

Expectedval = num/M
print("Expected value:")
print(Expectedval)

from scipy import stats
chisqrd = np.sum(((n-Expectedval)**2)/Expectedval)
c = stats.chi2.cdf(chisqrd,10)
chipdf = 1 - c
print('Chi-squared:')
print(chipdf)


# # Q3 part c

# In[20]:


#create new sample
num= 1000
M=10
new_samples = np.random.randint(1,11,num)
n1, bins, patches = plt.hist(new_samples, bins=10, color='red',
                            alpha=0.5, rwidth = 0.9)
plt.grid(axis='y', alpha=0.5)
plt.xlabel('Samples')
plt.ylabel('Output')
plt.title('Histogram')

#find observed and expected values
print('Observed values:')
print(n1)

Expectedval = num/M
print("Expected value:")
print(Expectedval)

from scipy import stats
new_chisqrd = np.sum(((n1-Expectedval)**2)/Expectedval)
c1 = stats.chi2.cdf(new_chisqrd,10)
new_chipdf = 1 - c1
print('Chi-squared New:')
print(new_chipdf)


# In[ ]:





# In[ ]:




