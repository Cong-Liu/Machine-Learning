
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from pylab import*
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

def readfile(filename = "svar-set1.dat"):
    data = pd.read_csv(filename,skiprows = 5, sep =" ", skipinitialspace=True, header = None)
    datafile = []
    datafile = np.asarray(data)
    s = np.split(datafile, [1], axis=1)
#     s = np.split(datafile, shape[1]-1, axis=1)
    x=s[0]
    y=s[1]
    return x,y



# In[3]:

# K fold
def k_fold(size,k, shuffle = False):
    index = np.arange(0,size) 
    if shuffle:
        np.random.shuffle(index)
    index = np.reshape(index, (size/k,k))

    test = []     
    train = []
    cursor = 0
    for i in range(0,size/k):
        test.append(index[i])
        temp = np.delete(index,i,0)
        train.append(temp.flatten())
    return test, train


# In[4]:

# polynomail
#return formed z = [1,x]
def trans_X(x,factor):
    z = np.ones(len(x))
    for i in range(1,factor+1):
        temp = np.power(x,i)
        z = np.c_[z,temp]
    return z



# In[5]:

#pridict
def prediction(x, y, fited = False, getTheta = False):
    z = x 
    if fited is False:
        z = np.insert(x, 0, 1, axis=1)
      
    theta = dot(pinv(z),y)
    if getTheta:
        return theta
    pre = dot(z,theta)
    return pre


def getTest_ERR(x_test, y_test, theta):
    z = np.insert(x_test, 0, 1, axis=1)
    y_pre = dot(z,theta)
    m = (y_pre-y_test)
    d_MSE = np.sum(np.power(m,2))/m.size
    d_RSE = np.sum((np.power(m,2)) / np.square(y_test))/m.size
    return d_MSE, d_RSE


# In[6]:

# compute MSE & RSE

def data_ERR(x,y):
    pre = prediction(x,y)
    m = (pre-y)
    d_MSE = np.sum(np.power(m,2))/m.size
    d_RSE = np.sum((np.power(m,2)) / np.square(y))/m.size
    return d_MSE, d_RSE


# In[7]:

#draw a picture

def draw(x,y,c = 'b.', title = None, xlabel= None, ylabel= None):
    
    plt.plot(x, y, c)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# In[8]:


def is_K_Fold(x,y,k = 10, shuffle = False):

    i_test, i_train = k_fold(len(x),k, shuffle)
    mse_tr = 0;
    rse_tr = 0
    mse_ts = 0
    rse_ts = 0

    for i in range(0,k):
        
        x_test = x[i_test[i]]
        y_test = y[i_test[i]]
        x_train = x[i_train[i]]
        y_train = y[i_train[i]]


        m_tr, r_tr = data_ERR(x_train, y_train)

        mse_tr+= m_tr
        rse_tr +=r_tr

        theta = prediction(x_train, y_train, getTheta = True)
        m_ts, r_ts = getTest_ERR(x_test, y_test,theta)

        mse_ts += m_ts
        rse_ts += r_ts
        
    return (mse_tr, rse_tr, mse_ts, rse_ts)


# In[9]:


def reduceData(x,y,part = 1.0):
    size = len(x)
    cursor = int(size/10)
    new_size = int(size*part+1)
    
 
    re_x = x[:new_size]
    re_y = y[:new_size]
    

    x_test = re_x[:cursor]
    y_test = re_y[:cursor]
    
    x_train = re_x[cursor:]
    y_train = re_y[cursor:]
    
    m_tr, r_tr = data_ERR(x_train, y_train)
    theta = prediction(x_train, y_train, getTheta = True)
    m_ts, r_ts = getTest_ERR(x_test, y_test,theta)

    return (r_tr, r_ts)



# In[10]:

# draw a form of reduced data errors.
def d_reduce(x,y,is_reduce,fn = None):


    n = int(is_reduce*10)
    L_rslt = np.zeros((2,n))
    # do reduce n times
    for j in range(1,n):
        rslt = reduceData(x,y,double(j)/10)
        L_rslt[0,j] = rslt[0]
        L_rslt[1,j] = rslt[1]

    lbl = np.arange(n)
    plt.plot(lbl, L_rslt[0], 'b-')
    plt.plot(lbl, L_rslt[1],'r-')
    plt.title(fn)
    plt.xlabel('Trainning error in blue, testing error in red')
    plt.ylabel('RSE')
    plt.show()


# In[11]:

def CrossValidation(filename = "svar-set1.dat", k = 0, printLinear = False, fit = 1, shuffle = False, is_reduce = 1.0, flg = False, ftrs = None, lbls = None):
      
    x,y = readfile(filename)

    if (is_reduce < 1.0):
        d_reduce(x,y,is_reduce) 

    if printLinear: 
        pre = prediction(x,y)
        plt.plot(x, y, 'b.')
        plt.title("Dataset:"+filename)
        plt.plot(x, pre,'r.')
        plt.show()    
    
    if k is not 0:
        err = is_K_Fold(x,y,k, shuffle)
        print 'Average training MSE is', err[0]/10
        print 'Average training RSE is', err[1]/10
        print 'Average testing MSE is', err[2]/10
        print 'Average testing RSE is', err[3]/10
    
    if (fit>1) :
        fit = int(fit)
        if flg:
            x = ftrs
            y = lbls
            L_MSE = np.zeros(fit)
        L_RSE = np.zeros(fit)
        for i in range(1,fit):
            z = trans_X(x,i)
            pre_P = prediction(z,y,fited = True)
            x_fit = z[:,1:]
            err_M, err_R = data_ERR(x_fit,y)
            L_RSE[i] = err_R
            L_MSE[i] = err_M
        n = np.arange(fit)
        if not flg:
            print ''
            draw(x=n,y=L_RSE,c = 'r.-', title = filename, xlabel = 'Factors of features', ylabel = 'RSE')
        if flg:
            print ''
            draw(x=n,y=L_MSE,c = 'b.-', title = filename, xlabel = 'Factors of features', ylabel = 'MSE')


# In[ ]:



