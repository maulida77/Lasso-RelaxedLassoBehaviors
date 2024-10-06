import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from sklearn import linear_model
from sklearn import datasets
import math
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import cross_val_score
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import host_subplot
import seaborn as sns
import time


begin = time.time()



def get_numpy_data(data, features):
    feature_matrix=data[features].to_numpy()
    return (feature_matrix)

def normalize_features(simple_feature_matrix):
    m,n = simple_feature_matrix.shape
    normalize_feature_matrix=(simple_feature_matrix-simple_feature_matrix.mean(axis=0))/(simple_feature_matrix.std(axis=0) ) #*math.sqrt(m/(m-1))
    std = simple_feature_matrix.std(axis=0)
    return normalize_feature_matrix, std

def centered_output(output):
    center_output=output-output.mean()
    return center_output

def predict_output(feature_matrix, weights, betanol):
    predictions=betanol+np.dot(feature_matrix,weights)
    return predictions

def soft_threshold(rho,lamda):
    '''Soft threshold function used for normalized data and lasso regression'''
    if rho < - lamda:
        return (rho + lamda)
    elif rho >  lamda:
        return (rho - lamda)
    else: 
        return 0
    
def coordinate_descent_lasso(theta,X,y,lamda, tol):
    '''Coordinate gradient descent for lasso regression - for normalized data. 
    The intercept parameter allows to specify whether or not we regularize theta_0'''
    
    #Initialisation of useful values 
    m,n = X.shape 
    #Looping until max number of iterations
    while True: 
        #Looping through each coordinate
        change=np.zeros((n,1))
        for j in range(n):            
            #Vectorized implementation
            old_theta_j=theta[j]
            X_j = X[:,j].reshape(-1,1)
            y_pred=np.dot(X,theta)
            rho=1/m*np.sum(X_j*(y - y_pred  + theta[j]*X_j))
            weight_j =  soft_threshold(rho, lamda)
            change[j]=abs(weight_j-old_theta_j)
            theta[j]=weight_j
        if np.all(abs(change) < tol):
            break
    return theta.flatten()

def get_MSE(y, y_pred):
    # RMSE = math.sqrt(np.square(np.subtract(y,y_pred)).mean())
    MSE = np.square(np.subtract(y,y_pred)).mean()
    return(MSE)

def get_Rsquare(y, y_pred):
    ybar=y.mean()
    # Rsquare = (np.square(np.subtract(y_pred, ybar)).sum())/(np.square(np.subtract(y,ybar)).sum()) 
    Rsquare = 1-(np.square(np.subtract(y,y_pred)).sum())/(np.square(np.subtract(y,ybar)).sum())   
    return(Rsquare)

def get_Rsquare_adj(y, y_pred, m,n):
    ybar=y.mean()
    Rsquare_adj = 1-((np.square(np.subtract(y,y_pred)).sum())/(m-n))/((np.square(np.subtract(y,ybar)).sum())/(m-1))
    return(Rsquare_adj)

def theta_LS_full (theta_lasso, trainvalid_data_standardized, trainvalid_y_centered):
    m,n = trainvalid_data_standardized.shape
    theta_LS_full=np.zeros(n)
    if (np.count_nonzero(theta_lasso) > 0):
        idx= np.where(theta_lasso != 0)[0]
        X_Active = trainvalid_data_standardized[:,idx]
        lin = linear_model.LinearRegression()
        lin.fit(X_Active, trainvalid_y_centered)
        theta_LS= lin.coef_
        theta_LS_full[idx]= theta_LS
    return(theta_LS_full)

def k_fold_cv_relaxo(k, lamda, gamma, trainvalid_data, trainvalid_y, tol):
    m,n = trainvalid_data.shape
    val_error=np.zeros((k,1))
    for i in range(k):
        start = (m*i)//k
        end = (m*(i+1))//k-1
        valid_data =trainvalid_data[start:end+1]
        valid_y = trainvalid_y[start:end+1]
        train_data=np.vstack([trainvalid_data[end+1:m],trainvalid_data[0:start]])
        train_y=np.vstack([trainvalid_y[end+1:m],trainvalid_y[0:start]])
        initial_theta= np.ones((n,1))
        train_data_standardized, std = normalize_features(train_data)
        train_y_centered = centered_output(train_y)
        theta_lasso = coordinate_descent_lasso(initial_theta,train_data_standardized,train_y_centered, lamda,tol)
        theta_lasso_ori = theta_lasso/std
        theta_LSf=np.zeros(n)
        if gamma!=1 :
            theta_LSf=theta_LS_full (theta_lasso,  train_data_standardized,train_y_centered,)
        theta_relaxo= gamma*theta_lasso+(1-gamma)*theta_LSf
        theta_relaxo_ori = theta_relaxo/std
        # y_pred = np.dot(valid_data,theta_relaxo_ori).reshape(-1,1)
        betanol=train_y.mean()-np.dot(train_data.mean(axis=0),theta_relaxo_ori)
        y_pred = predict_output(valid_data,theta_relaxo_ori, betanol).reshape(-1,1)
        val_error[i]=get_MSE(valid_y, y_pred)
    Av_val_error=val_error.mean()
    return Av_val_error

def Relaxo_best_param(lamda, gamma, initial_theta,trainvalid_data, trainvalid_y, tol, k):
    m,n = trainvalid_data.shape
    trainvalid_data_standardized, std_tv = normalize_features(trainvalid_data)
    trainvalid_y_centered = centered_output(trainvalid_y)
    result=np.zeros((len(lamda)*len(gamma),5+n))
    i=0
    for l in lamda :
        theta_lasso=coordinate_descent_lasso(initial_theta,trainvalid_data_standardized, trainvalid_y_centered,l, tol) #thetalasso_tv
        theta_LS=theta_LS_full (theta_lasso, trainvalid_data_standardized, trainvalid_y_centered)
        for t in gamma: 
            result[i, 0]= l #lamda
            result[i, 1]=t #gamma
            result[i, 2]=k_fold_cv_relaxo(k, l, t,  trainvalid_data, trainvalid_y, tol) #cv_mse
            theta_LSf=np.zeros(n)
            if (t!=1) :
                theta_LSf=theta_LS
            result[i, 3]=t*(trainvalid_y.mean()-np.dot(trainvalid_data.mean(axis=0),theta_lasso/std_tv))+(1-t)*(trainvalid_y.mean()-np.dot(trainvalid_data.mean(axis=0),theta_LSf/std_tv)) #thetanol
            result[i, 4:4+n]=(t*theta_lasso+(1-t)*theta_LSf)/std_tv #ori_thetarelaxo_tv
            result[i, 4+n]=np.count_nonzero(result[i, 4:4+n]) #number non zero_tv  
            # initial_theta=result[i, 4:4+n].reshape(-1,1)   
            i=i+1
        initial_theta=theta_lasso.reshape(-1,1)
    best=result[np.argmin(result[:,2],0),:]
    return(result,best)



# ##################INPUT##########################
path_data = "Result/pollutan_0.5/"  #"Result/30_it_100_lambda/"
data=pd.DataFrame(pd.read_csv('pollutan.csv'))
features=['NOx','SO₂','CO','OC','NMVOC','BC','NH₃']
output=['Y']
X_ori=get_numpy_data(data, features)
X=get_numpy_data(data, features)/100000
y=get_numpy_data(data, output)
X_standardized, std = normalize_features(X)
y_centered = centered_output(y)
t_size=0.5
trainvalid_data,test_data = train_test_split(X, test_size=t_size, random_state=0)
trainvalid_y,test_y = train_test_split(y, test_size=t_size, random_state=0)
trainvalid_data_standardized, std_tv = normalize_features(trainvalid_data)
trainvalid_y_centered = centered_output(trainvalid_y)
m,n = trainvalid_data_standardized.shape
m1,n1 = test_data.shape
Lmax = np.max(np.dot(trainvalid_y_centered.T,trainvalid_data_standardized)/m)  #the least value lambda that make all coefficient zeros.   
l=math.ceil(math.log(Lmax,10))
lamda = np.logspace(l,-2,100)
# lamda = np.linspace(Lmax,0,100)
initial_theta = np.ones((n,1))
tol=1e-2
gamma=np.linspace(0,1,11)
fold=5
Coeff = ['$\\beta_1$', '$\\beta_2$', '$\\beta_3$', '$\\beta_4$', '$\\beta_5$', '$\\beta_6$', '$\\beta_7$']                        
featureslabel = ['$X_1$', '$X_2$', '$X_3$', '$X_4$', '$X_5$', '$X_6$', '$X_7$']
bt=50 #number of iteration for bootstrap



##################FIGURE 1 - Scatter Plot ##########################
fp=['SO₂','OC']
Xp=get_numpy_data(data, fp)/100000
yp=get_numpy_data(data, output)
fig, axs = plt.subplots(1,2, figsize=(4, 2))
plt.subplots_adjust(left=0.125, right=0.9, bottom = 0.1, top = 0.9, wspace=0.3, hspace=0.4)
for i, ax in enumerate(axs.flatten()):
    ax.scatter(Xp[:,i], yp, s=5 , c='g')
    ax.set(xlabel=fp[i],ylabel="Death rate") #Death rate from air pollution
    ax.xaxis.get_label().set_fontsize(8)
    ax.yaxis.get_label().set_fontsize(8)
    ax.tick_params(axis='both',labelsize=8)
plt.tight_layout()
plt.savefig(path_data+'img1_p.png') 
plt.close()



#################FIGURE 2 - CORRELATION MATRIX - Heat MAP##################
fig, axs = plt.subplots(1,1, figsize=(5, 3))
plt.subplots_adjust(left=0.1, right=1, bottom = 0.1, top = 0.95, wspace=0, hspace=0)
dataset = data[features]
matrix = dataset.corr()
hm=sns.heatmap(matrix,  annot=True, ax=axs, annot_kws={"fontsize":6}) # xticklabels=[4,8,16,32,64,128],yticklabels=[2,4,6,8]
cax = hm.figure.axes[-1]
cax.tick_params(labelsize=6)
plt.tick_params(axis='both',labelsize=6)
plt.tight_layout()
plt.savefig(path_data+'img2_heatmap_p.png') 
plt.close()



##################FIGURE 3 LASSO COEFFICIENT PATH- ##########################
theta_list = list()
# Run lasso regression for each lambda
for l in lamda:
    theta = coordinate_descent_lasso(initial_theta,trainvalid_data_standardized, trainvalid_y_centered,lamda=l, tol=1e-2)#/std
    theta_list.append(theta)
theta_lasso = np.stack(theta_list)
plt.figure(figsize = (6,4))
for i in range(n):
    plt.plot(lamda, theta_lasso[:,i], label = features[i])
plt.xscale('log')
plt.xlabel('Log($\\lambda$)',fontsize=8)
plt.ylabel('Coefficients',fontsize=8)
plt.legend(fontsize=6)
plt.tick_params(axis='both',labelsize=8 )
plt.axis('tight')
plt.savefig(path_data+'img3_p.png')
plt.close()



##################Calculate CVMSE for Best Lambda, Gamma and Saving Relaxo Coeff and NNZ for each lambda_gamma value##########################
AECV_lambda, best_cvmse=Relaxo_best_param(lamda, gamma, initial_theta,trainvalid_data, trainvalid_y, tol, fold)  
np.savetxt(path_data+"best_cvmse_0.csv", best_cvmse.T, delimiter=",")
np.savetxt(path_data+"AECV_lambda_0.csv", AECV_lambda, delimiter=",")




##################Best CV_MSE & Parameter For Certain NNZ & ITS MSE & PMSE RELAXO and LASSO ##########################
nnz_vector=np.unique(AECV_lambda[:, 4+n]).astype(int)
keeprelaxo=np.zeros((len(nnz_vector),11+n))
k=0
for i in nnz_vector:
    A= AECV_lambda[AECV_lambda[:,4+n]==i][:,:]
    keeprelaxo[k,0:n+5]=A[np.argmin(A[:,2],0),:] #lamdamin, cvmse 
    # y_tv_pred = np.dot(trainvalid_data,keeprelaxo[k,3:n+3]).reshape(-1,1)
    y_tv_pred = predict_output(trainvalid_data,keeprelaxo[k,4:n+4], keeprelaxo[k,3]).reshape(-1,1)
    keeprelaxo[k,n+5]= get_MSE(trainvalid_y, y_tv_pred) #mse tv
    y_test_pred = predict_output(test_data,keeprelaxo[k,4:n+4], keeprelaxo[k,3]).reshape(-1,1)
    # y_test_pred = np.dot(test_data,keeprelaxo[k,4:n+4]).reshape(-1,1)
    keeprelaxo[k,n+6]= get_MSE(test_y, y_test_pred) #mse tes->pmse
    keeprelaxo[k,n+7]=get_Rsquare(trainvalid_y, y_tv_pred)# Rsquare tv
    keeprelaxo[k,n+8]=get_Rsquare(test_y, y_test_pred)# Rsquare test set
    keeprelaxo[k,n+9]=get_Rsquare_adj(trainvalid_y, y_tv_pred,m,keeprelaxo[k,n+4])# Rsquare_adj tv
    keeprelaxo[k,n+10]=get_Rsquare_adj(test_y, y_test_pred,m1,keeprelaxo[k,n+4])# Rsquare_adj test set
    k=k+1
np.savetxt(path_data+"keeprelaxo_0.csv", keeprelaxo, delimiter=",") 
lassoest=AECV_lambda[AECV_lambda[:,1]==1][:,:]
keeplasso=np.zeros((len(nnz_vector),11+n))
k=0
for i in nnz_vector:
    B= lassoest[lassoest[:,4+n]==i][:,:]
    keeplasso[k,0:n+5]=B[np.argmin(B[:,2],0),:] #lamdamin, cvmse 
    # y_tv_pred = np.dot(trainvalid_data,keeplasso[k,3:n+3]).reshape(-1,1)
    y_tv_pred = predict_output(trainvalid_data,keeplasso[k,4:n+4], keeplasso[k,3]).reshape(-1,1)
    keeplasso[k,n+5]= get_MSE(trainvalid_y, y_tv_pred) #mse tv
    # y_test_pred = np.dot(test_data,keeplasso[k,3:n+3]).reshape(-1,1)
    y_test_pred = predict_output(test_data,keeplasso[k,4:n+4], keeplasso[k,3]).reshape(-1,1)
    keeplasso[k,n+6]= get_MSE(test_y, y_test_pred) #mse tes->pmse
    keeplasso[k,n+7]=get_Rsquare(trainvalid_y, y_tv_pred)# Rsquare tv
    keeplasso[k,n+8]=get_Rsquare(test_y, y_test_pred)# Rsquare test set
    keeplasso[k,n+9]=get_Rsquare_adj(trainvalid_y, y_tv_pred,m,keeplasso[k,n+4])# Rsquare_adj tv
    keeplasso[k,n+10]=get_Rsquare_adj(test_y, y_test_pred,m1,keeplasso[k,n+4])# Rsquare_adj test set
    k=k+1
np.savetxt(path_data+"keeplasso_0.csv", keeplasso, delimiter=",") 
bestlasso_0=keeplasso[np.argmin(keeplasso[:,2],0),:]
np.savetxt(path_data+"bestlasso_0.csv", bestlasso_0, delimiter=",") 



##################FIGURE 4 Relaxo COEFFICIENT PATH, best gamma- ##########################
theta_list = list()
# Run relaxo regression for each lambda, best gamma
t=best_cvmse[1]
print(t)
for l in lamda:
    # theta = coordinate_descent_lasso(initial_theta,X_standardized,y_centered,lamda=l, tol=1e-2)/std
    theta_lasso = coordinate_descent_lasso(initial_theta,trainvalid_data_standardized, trainvalid_y_centered,lamda=l, tol=1e-2)
    theta_LSf=theta_LS_full (theta_lasso, trainvalid_data_standardized, trainvalid_y_centered)
    theta_relaxo=(t*theta_lasso+(1-t)*theta_LSf)# /std for ori_thetarelaxo_tv
    theta_list.append(theta_relaxo)
theta_relaxo = np.stack(theta_list)
plt.figure(figsize = (6,4))
for i in range(n):
    plt.plot(lamda, theta_relaxo[:,i], label = features[i])
plt.xscale('log')
plt.xlabel('Log($\\lambda$)',fontsize=8)
plt.ylabel('Coefficients ($\\gamma=0.1$)', fontsize=8)
plt.legend(fontsize=6)
plt.tick_params(axis='both',labelsize=8 )
plt.axis('tight')
plt.savefig(path_data+'img4_p.png')
plt.close()



##################FIGURE 5 Average MSE Cross Validation Relaxo- ##########################
color1 = ['black', 'gold', 'tomato', 'darkkhaki', 'darkseagreen', 'aquamarine', 'dodgerblue', 'plum', 'blue', 
'green', 'red', 'cyan', 'lime', 'indigo', 'fuchsia', 'navy', 'pink', 'orange','violet','silver','teal']
plt.figure(figsize = (6,4))
ax = host_subplot(111)
ax2 = ax.twin()  # ax2 is responsible for "top" axis and "right" axis
j=0
for t in gamma :
    plt.plot(AECV_lambda[AECV_lambda[:,1]==t][:,0], AECV_lambda[AECV_lambda[:,1]==t][:,2], '--', color=color1[j],lw=0.5, ms=2, label=round(t, 1))
    j=j+1
plt.plot(best_cvmse[0], best_cvmse[2], 'x', color='red', ms=3)
plt.plot(bestlasso_0[0], bestlasso_0[2], 'x', color='black', ms=3)
plt.tick_params(axis='both',labelsize=8)  
plt.xscale('log')
plt.xlabel('Log($\\lambda$)', fontsize=8)
plt.ylabel('CV Mean Squared Error',fontsize=8)
plt.legend(title = "$\\gamma$", fontsize = 6, title_fontsize=6)   
ax2.set_xticks(keeplasso[:,0],labels=keeplasso[:,n+4].astype(int))
ax2.set_yticks([],color='w')
ax2.tick_params(axis='both',labelsize=6)
ax2.set_xlabel('Number Non Zero Coefficient', fontsize=6)
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.axis["top"].major_ticklabels.set_visible(True)
plt.savefig(path_data+'img5_p.png')
plt.close()



####RANK of Train Valid Data#########
I=np.zeros((50,2)) # cekranktvdata
for i in range(50):
    trainvalid_data,test_data = train_test_split(X, test_size=t_size, random_state=i)
    I[i, :] = [np.linalg.matrix_rank(trainvalid_data),np.linalg.matrix_rank(test_data)]
np.savetxt(path_data+"ranktvdata.csv", I, delimiter=",") 



###################################BOOTSTRAP###########################################
BEST = np.zeros((bt,12+n))
# ols =np.zeros((bt,3+n))
keeprelaxo = np.zeros(((n+1)*bt,12+n))
keeplasso = np.zeros(((n+1)*bt,12+n))
bestlasso=np.zeros((bt,12+n))
sort_features=np.zeros((bt,n))
r=0
s=0
for rdm in range(bt):
    print(rdm)
    trainvalid_data,test_data = train_test_split(X, test_size=t_size, random_state=rdm)
    trainvalid_y,test_y = train_test_split(y, test_size=t_size, random_state=rdm)
    trainvalid_data_standardized, std_tv = normalize_features(trainvalid_data)
    trainvalid_y_centered = centered_output(trainvalid_y)
    Lmax = np.max(np.dot(trainvalid_y_centered.T,trainvalid_data_standardized)/m)  #the least value lambda that make all coefficient zeros.   
    l=math.ceil(math.log(Lmax,10))
    lamda = np.logspace(l,-2,100)
    ##################Calculate CVMSE for Best Lambda, Gamma and Saving Relaxo Coeff and NNZ for each lambda_gamma value##########################
    AECV_lambda, best_cvmse=Relaxo_best_param(lamda, gamma, initial_theta,trainvalid_data, trainvalid_y, tol, fold)    
    print(AECV_lambda[:, 4+n].flatten())
    BEST[rdm,0]=rdm
    BEST[rdm,1:6+n] = best_cvmse
    # y_tv_pred = np.dot(trainvalid_data,best_cvmse[4:n+4]).reshape(-1,1)
    y_tv_pred = predict_output(trainvalid_data,best_cvmse[4:n+4], best_cvmse[3]).reshape(-1,1)
    BEST[rdm,6+n]= get_MSE(trainvalid_y, y_tv_pred) #mse tv
    # y_test_pred = np.dot(test_data,best_cvmse[4:n+4]).reshape(-1,1)
    y_test_pred = predict_output(test_data,best_cvmse[4:n+4], best_cvmse[3]).reshape(-1,1)
    BEST[rdm,7+n]= get_MSE(test_y, y_test_pred) #mse tes->pmse
    BEST[rdm,8+n]=get_Rsquare(trainvalid_y, y_tv_pred)# Rsquare tv
    BEST[rdm,9+n]=get_Rsquare(test_y, y_test_pred)# Rsquare test set
    BEST[rdm,10+n]=get_Rsquare_adj(trainvalid_y, y_tv_pred,m,BEST[rdm,5+n])# Rsquare_adj tv
    BEST[rdm,11+n]=get_Rsquare_adj(test_y, y_test_pred,m1,BEST[rdm,5+n])# Rsquare_adj test set
    ##################Best CV_MSE & Parameter For Certain NNZ & ITS MSE & PMSE RELAXO and LASSO ##########################
    nnz_vector=np.unique(AECV_lambda[:, 4+n]).astype(int)
    print(nnz_vector)
    # keeprelaxo=np.zeros((len(nnz_vector)*bt,7+n))
    for i in nnz_vector:
        keeprelaxo[r,0] =rdm
        A= AECV_lambda[AECV_lambda[:,4+n]==i][:,:]
        keeprelaxo[r,1:n+6]=A[np.argmin(A[:,2],0),:] #lamdamin, cvmse 
        # y_tv_pred = np.dot(trainvalid_data,keeprelaxo[r,5:n+5]).reshape(-1,1)
        y_tv_pred = predict_output(trainvalid_data,keeprelaxo[r,5:n+5], keeprelaxo[r,4]).reshape(-1,1)
        keeprelaxo[r,n+6]= get_MSE(trainvalid_y, y_tv_pred) #mse tv
        # y_test_pred = np.dot(test_data,keeprelaxo[r,5:n+5]).reshape(-1,1)
        y_test_pred =predict_output(test_data,keeprelaxo[r,5:n+5], keeprelaxo[r,4]).reshape(-1,1)
        keeprelaxo[r,n+7]= get_MSE(test_y, y_test_pred) #mse tes->pmse
        keeprelaxo[r,8+n]=get_Rsquare(trainvalid_y, y_tv_pred)# Rsquare tv
        keeprelaxo[r,9+n]=get_Rsquare(test_y, y_test_pred)# Rsquare test set
        keeprelaxo[r,10+n]=get_Rsquare_adj(trainvalid_y, y_tv_pred,m,keeprelaxo[r,n+5])# Rsquare_adj tv
        keeprelaxo[r,11+n]=get_Rsquare_adj(test_y, y_test_pred,m1,keeprelaxo[r,n+5])# Rsquare_adj test set
        r=r+1  
    a =  np.count_nonzero(keeprelaxo[keeprelaxo[:,0]==rdm,5:5+n],0)     # count nnz for most important value
    sort_features[rdm]=a
    lassoest=AECV_lambda[AECV_lambda[:,1]==1][:,:]
    # keeplasso=np.zeros((len(nnz_vector)*bt,7+n))
    for i in nnz_vector:
        keeplasso[s,0] =rdm
        B= lassoest[lassoest[:,4+n]==i][:,:]
        keeplasso[s,1:n+6]=B[np.argmin(B[:,2],0),:] #lamdamin, cvmse 
        # y_tv_pred = np.dot(trainvalid_data,keeplasso[s,5:n+5]).reshape(-1,1)
        y_tv_pred = predict_output(trainvalid_data,keeplasso[s,5:n+5], keeplasso[s,4]).reshape(-1,1)
        keeplasso[s,n+6]= get_MSE(trainvalid_y, y_tv_pred) #mse tv
        # y_test_pred = np.dot(test_data,keeplasso[s,5:n+5]).reshape(-1,1)
        y_test_pred = predict_output(test_data,keeplasso[s,5:n+5], keeplasso[s,4]).reshape(-1,1)
        keeplasso[s,n+7]= get_MSE(test_y, y_test_pred) #mse tes->pmse
        keeplasso[s,8+n]=get_Rsquare(trainvalid_y, y_tv_pred)# Rsquare tv
        keeplasso[s,9+n]=get_Rsquare(test_y, y_test_pred)# Rsquare test set
        keeplasso[s,10+n]=get_Rsquare_adj(trainvalid_y, y_tv_pred,m,keeplasso[s,n+5])# Rsquare_adj tv
        keeplasso[s,11+n]=get_Rsquare_adj(test_y, y_test_pred,m1,keeplasso[s,n+5])# Rsquare_adj test set
        s=s+1
    T=keeplasso[0:s,:]
    T_T=T[T[:,0]==rdm,:]
    bestlasso[rdm, :]=T_T[np.argmin(T_T[:,3],0),:]
    # ol = linear_model.LinearRegression()
    # ol.fit( trainvalid_data_standardized, trainvalid_y_centered)
    # ols[rdm, 0:n]= ol.coef_/std_tv
    # ols[rdm, n]=np.count_nonzero(ols[rdm, 0:n])
    # y_tv_pred = np.dot(trainvalid_data, ols[rdm, 0:n]).reshape(-1,1)
    # ols[rdm, n+1]= get_MSE(trainvalid_y, y_tv_pred) #mse tv
    # y_test_pred = np.dot(test_data, ols[rdm, 0:n]).reshape(-1,1)
    # ols[rdm, n+2]= get_MSE(test_y, y_test_pred) #mse tes->pmse
keep_relaxo = keeprelaxo [0:r,:]
keep_lasso = keeplasso [0:s,:] 
np.savetxt(path_data+"BEST.csv", BEST, delimiter=",")
np.savetxt(path_data+"bootstrap_keep_relaxo.csv", keep_relaxo, delimiter=",")
np.savetxt(path_data+"bootstrap_keep_lasso.csv", keep_lasso, delimiter=",")
# np.savetxt(path_data+"bootstrap_ols.csv", ols, delimiter=",")
np.savetxt(path_data+"bootstrap_sort_features.csv", sort_features, delimiter=",")
np.savetxt(path_data+"bootstrap_bestlasso.csv", bestlasso, delimiter=",")

nnz_vector=np.unique(keep_relaxo[:, 5+n]).astype(int)
Save_result_relaxo = np.zeros((len(nnz_vector),13))
Save_result_lasso = np.zeros((len(nnz_vector),13))
j=0
for i in nnz_vector:
    Save_result_relaxo[j, 0]= i
    Save_result_relaxo[j, 1]= keep_relaxo[keep_relaxo[:, n+5]==i, n+6].mean(axis=0) # mean mse tv relaxo
    Save_result_relaxo[j, 2]= keep_relaxo[keep_relaxo[:, n+5]==i, n+6].std(axis=0) # std mse tv relaxo
    Save_result_relaxo[j, 3]= keep_relaxo[keep_relaxo[:, n+5]==i, n+7].mean(axis=0) # mean pmse  relaxo
    Save_result_relaxo[j, 4]= keep_relaxo[keep_relaxo[:, n+5]==i, n+7].std(axis=0) # std pmse  relaxo
    Save_result_relaxo[j, 5]= keep_relaxo[keep_relaxo[:, n+5]==i, n+8].mean(axis=0) # mean R2 tv relaxo
    Save_result_relaxo[j, 6]= keep_relaxo[keep_relaxo[:, n+5]==i, n+8].std(axis=0) # std R2 tv relaxo
    Save_result_relaxo[j, 7]= keep_relaxo[keep_relaxo[:, n+5]==i, n+9].mean(axis=0) # mean r2test relaxo
    Save_result_relaxo[j, 8]= keep_relaxo[keep_relaxo[:, n+5]==i, n+9].std(axis=0) # std r2test  relaxo
    Save_result_relaxo[j, 9]= keep_relaxo[keep_relaxo[:, n+5]==i, n+10].mean(axis=0) # mean adjR2 tv relaxo
    Save_result_relaxo[j, 10]= keep_relaxo[keep_relaxo[:, n+5]==i, n+10].std(axis=0) # std adjR2 tv relaxo
    Save_result_relaxo[j, 11]= keep_relaxo[keep_relaxo[:, n+5]==i, n+11].mean(axis=0) # mean adjr2test relaxo
    Save_result_relaxo[j, 12]= keep_relaxo[keep_relaxo[:, n+5]==i, n+11].std(axis=0) # std adjr2test  relaxo

    Save_result_lasso[j, 0]= i
    Save_result_lasso[j, 1]= keep_lasso[keep_lasso[:, n+5]==i, n+6].mean(axis=0) # mean mse tv lasso
    Save_result_lasso[j, 2]= keep_lasso[keep_lasso[:, n+5]==i, n+6].std(axis=0) # std mse tv lasso
    Save_result_lasso[j, 3]= keep_lasso[keep_lasso[:, n+5]==i, n+7].mean(axis=0) # mean pmse  lasso
    Save_result_lasso[j, 4]= keep_lasso[keep_lasso[:, n+5]==i, n+7].std(axis=0) # std pmse lasso
    Save_result_lasso[j, 5]= keep_lasso[keep_lasso[:, n+5]==i, n+8].mean(axis=0) # mean R2 tv lasso
    Save_result_lasso[j, 6]= keep_lasso[keep_lasso[:, n+5]==i, n+8].std(axis=0) # std R2 tv lasso
    Save_result_lasso[j, 7]= keep_lasso[keep_lasso[:, n+5]==i, n+9].mean(axis=0) # mean r2test lasso
    Save_result_lasso[j, 8]= keep_lasso[keep_lasso[:, n+5]==i, n+9].std(axis=0) # std r2test  lasso
    Save_result_lasso[j, 9]= keep_lasso[keep_lasso[:, n+5]==i, n+10].mean(axis=0) # mean adjR2 tv lasso
    Save_result_lasso[j, 10]= keep_lasso[keep_lasso[:, n+5]==i, n+10].std(axis=0) # std adjR2 tv lasso
    Save_result_lasso[j, 11]= keep_lasso[keep_lasso[:, n+5]==i, n+11].mean(axis=0) # mean adjr2test lasso
    Save_result_lasso[j, 12]= keep_lasso[keep_lasso[:, n+5]==i, n+11].std(axis=0) # std adjr2test  lasso
    j=j+1
MeanSTD_Relaxo= [BEST[:,n+5].mean(axis=0), BEST[:,n+5].std(axis=0),BEST[:,3].mean(axis=0), BEST[:,3].std(axis=0), BEST[:,n+6].mean(axis=0), BEST[:,n+6].std(axis=0), BEST[:,n+7].mean(axis=0), BEST[:,n+7].std(axis=0), BEST[:,n+8].mean(axis=0), BEST[:,n+8].std(axis=0), BEST[:,n+9].mean(axis=0), BEST[:,n+9].std(axis=0), BEST[:,n+10].mean(axis=0), BEST[:,n+10].std(axis=0), BEST[:,n+11].mean(axis=0), BEST[:,n+11].std(axis=0)]
MeanSTD_Lasso= [bestlasso[:,n+5].mean(axis=0), bestlasso[:,n+5].std(axis=0),bestlasso[:,3].mean(axis=0), bestlasso[:,3].std(axis=0),bestlasso[:,n+6].mean(axis=0), bestlasso[:,n+6].std(axis=0), bestlasso[:,n+7].mean(axis=0), bestlasso[:,n+7].std(axis=0), bestlasso[:,n+8].mean(axis=0), bestlasso[:,n+8].std(axis=0), bestlasso[:,n+9].mean(axis=0), bestlasso[:,n+9].std(axis=0), bestlasso[:,n+10].mean(axis=0), bestlasso[:,n+10].std(axis=0), bestlasso[:,n+11].mean(axis=0), bestlasso[:,n+11].std(axis=0)]
np.savetxt(path_data+"Save_result_relaxo.csv", Save_result_relaxo, delimiter=",") 
np.savetxt(path_data+"Save_result_lasso.csv", Save_result_lasso, delimiter=",") 
np.savetxt(path_data+"MeanSTD_Relaxo.csv", MeanSTD_Relaxo, delimiter=",") 
np.savetxt(path_data+"MeanSTD_Lasso.csv", MeanSTD_Lasso, delimiter=",")                                                                          


#FIGURE 6 : Iteration Vs Features Rate
plt.figure(figsize = (6,4))                        
for t in range(n) :
    plt.plot(np.linspace(0,bt-1,bt),sort_features[:,t], 'o--', lw=0.5, ms=2, label=featureslabel[t])
plt.tick_params(axis='both',labelsize=8)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('Features Rate',fontsize=8)
plt.legend( fontsize = 6, loc='upper left')  
plt.savefig(path_data+'img6_p.png')   
plt.close()



#FIGURE 7 : Iteration Vs NNZ
plt.figure(figsize = (6,4))
plt.plot(BEST[:,0],BEST[:,n+5], 'o--', color='red',lw=0.5, ms=2, label='Relaxo')
plt.plot(bestlasso[:,0],bestlasso[:,n+5], 'o--', color='black',lw=0.5, ms=2, label='Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('nnz',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img7_p.png') 
plt.close()



# FIGURE 8: Histogram of NNZ
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(2.5, 4))
axes[1].hist(BEST[:, n+5].astype(int), color='red', edgecolor='black')
axes[1].set_xlabel('nnz (relaxo)')
axes[1].set_ylabel('Frequency')
axes[1].tick_params(axis='both',labelsize=8)
axes[1].yaxis.get_label().set_fontsize(8)
axes[1].xaxis.get_label().set_fontsize(8)
axes[0].hist(bestlasso[:, n+5].astype(int),  color='black', edgecolor='black')
axes[0].set_xlabel('nnz (lasso)')
axes[0].set_ylabel('Frequency')
axes[0].tick_params(axis='both',labelsize=8)
axes[0].yaxis.get_label().set_fontsize(8)
axes[0].xaxis.get_label().set_fontsize(8)
plt.tight_layout()
plt.savefig(path_data+'img8_p.png') 
plt.close()



##################FIGURE 9:  Box Plot##########################
df1 = pd.DataFrame(BEST[:, 5:5+n], columns=Coeff)
df2 = pd.DataFrame(bestlasso[:, 5:5+n], columns=Coeff)
plt.figure(figsize = (3,3.2))
bplot=df1.boxplot(column=Coeff, vert=False, patch_artist=True, boxprops = dict(facecolor = "red", color='red'),  whiskerprops=dict(color='black'), capprops=dict(color='black'),
 medianprops=dict(color='orange'), flierprops=dict(markerfacecolor='white', marker='o'))
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Coefficient (relaxo)', fontsize=8)
plt.grid(visible=None)
plt.savefig(path_data+'img9a_p.png') 
plt.figure(figsize = (3,3.2))
bplot=df2.boxplot(column=Coeff, vert=False, patch_artist=True,  boxprops = dict(facecolor = "black", color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'),
 medianprops=dict(color='orange'), flierprops=dict(markerfacecolor='white', marker='o'))
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Coefficient (lasso)', fontsize=8)
plt.grid(visible=None)
plt.savefig(path_data+'img9b_p.png') 
plt.close()



#FIGURE 10 : Iteration Vs Error
plt.figure(figsize = (6,4))
plt.plot(BEST[:,0],BEST[:,n+6], 'o-', color='red',lw=0.5, ms=2, label='MSE_Relaxo')
plt.plot(BEST[:,0],BEST[:,n+7], 'o--', color='red',lw=0.5, ms=2, label='PMSE_Relaxo')
plt.plot(BEST[:,0],BEST[:,3], 'o:', color='red',lw=0.5, ms=2, label='CVMSE_Relaxo')
plt.plot(bestlasso[:,0],bestlasso[:,n+6], 'o-', color='black',lw=0.5, ms=2, label='MSE_Lasso')
plt.plot(bestlasso[:,0],bestlasso[:,n+7], 'o--', color='black',lw=0.5, ms=2, label='PMSE_Lasso')
plt.plot(bestlasso[:,0],bestlasso[:,3], 'o:', color='black',lw=0.5, ms=2, label='CVMSE_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('Error Rate',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img10_p.png') 
plt.close()

plt.figure(figsize = (6,2.9))
plt.plot(BEST[:,0],BEST[:,n+6], 'o-', color='red',lw=0.5, ms=2, label='MSE_Relaxo')
plt.plot(bestlasso[:,0],bestlasso[:,n+6], 'o-', color='black',lw=0.5, ms=2, label='MSE_Lasso')
plt.tick_params(axis='both',labelsize=6)  
# plt.xscale('log')
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('MSE',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img10a_p.png') 
plt.close()

plt.figure(figsize = (6,2.9))
plt.plot(BEST[:,0],BEST[:,n+7], 'o--', color='red',lw=0.5, ms=2, label='PMSE_Relaxo')
plt.plot(bestlasso[:,0],bestlasso[:,n+7], 'o--', color='black',lw=0.5, ms=2, label='PMSE_Lasso')
plt.tick_params(axis='both',labelsize=6)  
# plt.xscale('log')
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('PMSE',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img10b_p.png') 
plt.close()

plt.figure(figsize = (6,2.9))
plt.plot(BEST[:,0],BEST[:,3], 'o:', color='red',lw=0.5, ms=2, label='CVMSE_Relaxo')
plt.plot(bestlasso[:,0],bestlasso[:,3], 'o:', color='black',lw=0.5, ms=2, label='CVMSE_Lasso')
plt.tick_params(axis='both',labelsize=6)   
# plt.xscale('log')
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('CVMSE',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img10c_p.png') 
plt.close()



#FIGURE 11 : Iteration Vs Rsquare
plt.figure(figsize = (6,4))
plt.plot(BEST[:,0],BEST[:,n+8], 'o-', color='red',lw=0.5, ms=2, label='$R^2$_Relaxo')
plt.plot(BEST[:,0],BEST[:,n+9], 'o--', color='red',lw=0.5, ms=2, label='P$R^2$_Relaxo')
plt.plot(bestlasso[:,0],bestlasso[:,n+8], 'o-', color='black',lw=0.5, ms=2, label='$R^2$_Lasso')
plt.plot(bestlasso[:,0],bestlasso[:,n+9], 'o--', color='black',lw=0.5, ms=2, label='P$R^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$R^2$',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img11a_p.png') 
plt.close()

plt.figure(figsize = (6,4))
plt.plot(BEST[:,0],BEST[:,n+8], 'o-', color='red',lw=0.5, ms=2, label='$R^2$_Relaxo')
plt.plot(bestlasso[:,0],bestlasso[:,n+8], 'o-', color='black',lw=0.5, ms=2, label='$R^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$R^2$',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img11b_p.png') 
plt.close()

plt.figure(figsize = (6,4))
plt.plot(BEST[:,0],BEST[:,n+9], 'o--', color='red',lw=0.5, ms=2, label='P$R^2$_Relaxo')
plt.plot(bestlasso[:,0],bestlasso[:,n+9], 'o--', color='black',lw=0.5, ms=2, label='P$R^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$PR^2$',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img11c_p.png') 
plt.close()

plt.figure(figsize = (6,4))
plt.plot(BEST[:,0],BEST[:,n+10], 'o-', color='red',lw=0.5, ms=2, label='$R^2$_Adj_Relaxo')
plt.plot(BEST[:,0],BEST[:,n+11], 'o--', color='red',lw=0.5, ms=2, label='P$R^2$_Adj_Relaxo')
plt.plot(bestlasso[:,0],bestlasso[:,n+10], 'o-', color='black',lw=0.5, ms=2, label='$R^2$_Adj_Lasso')
plt.plot(bestlasso[:,0],bestlasso[:,n+11], 'o--', color='black',lw=0.5, ms=2, label='P$R^2$_Adj_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('Adjusted $R^2$ ',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img11d_p.png') 
plt.close()

plt.figure(figsize = (6,4))
plt.plot(BEST[:,0],BEST[:,n+10], 'o-', color='red',lw=0.5, ms=2, label='$R^2$_Adj_Relaxo')
plt.plot(bestlasso[:,0],bestlasso[:,n+10], 'o-', color='black',lw=0.5, ms=2, label='$R^2$_Adj_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('Adjusted $R^2$',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img11e_p.png') 
plt.close()

plt.figure(figsize = (6,4))
plt.plot(BEST[:,0],BEST[:,n+11], 'o--', color='red',lw=0.5, ms=2, label='P$R^2$_Adj_Relaxo')
plt.plot(bestlasso[:,0],bestlasso[:,n+11], 'o--', color='black',lw=0.5, ms=2, label='P$R^2$_Adj_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('Adjusted $PR^2$ ',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img11f_p.png') 
plt.close()



#####FIGURE 12#####
###MSE###
plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==1][:,0],keeprelaxo[keeprelaxo[:,5+n]==1][:,n+6], 'o--', color='red',lw=0.5, ms=2, label='MSE_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==1][:,0],keeplasso[keeplasso[:,5+n]==1][:,n+6], 'o--', color='black',lw=0.5, ms=2, label='MSE_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$MSE$ (nnz=1)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12a1_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==2][:,0],keeprelaxo[keeprelaxo[:,5+n]==2][:,n+6], 'o--', color='red',lw=0.5, ms=2, label='MSE_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==2][:,0],keeplasso[keeplasso[:,5+n]==2][:,n+6], 'o--', color='black',lw=0.5, ms=2, label='MSE_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$MSE$ (nnz=2)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12a2_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==3][:,0],keeprelaxo[keeprelaxo[:,5+n]==3][:,n+6], 'o--', color='red',lw=0.5, ms=2, label='MSE_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==3][:,0],keeplasso[keeplasso[:,5+n]==3][:,n+6], 'o--', color='black',lw=0.5, ms=2, label='MSE_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$MSE$ (nnz=3)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12a3_p.png')
plt.close()

####PMSE######
plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==1][:,0],keeprelaxo[keeprelaxo[:,5+n]==1][:,n+7], 'o--', color='red',lw=0.5, ms=2, label='PMSE_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==1][:,0],keeplasso[keeplasso[:,5+n]==1][:,n+7], 'o--', color='black',lw=0.5, ms=2, label='PMSE_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$PMSE$ (nnz=1)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12b1_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==2][:,0],keeprelaxo[keeprelaxo[:,5+n]==2][:,n+7], 'o--', color='red',lw=0.5, ms=2, label='PMSE_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==2][:,0],keeplasso[keeplasso[:,5+n]==2][:,n+7], 'o--', color='black',lw=0.5, ms=2, label='PMSE_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$PMSE$ (nnz=2)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12b2_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==3][:,0],keeprelaxo[keeprelaxo[:,5+n]==3][:,n+7], 'o--', color='red',lw=0.5, ms=2, label='PMSE_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==3][:,0],keeplasso[keeplasso[:,5+n]==3][:,n+7], 'o--', color='black',lw=0.5, ms=2, label='PMSE_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$PMSE$ (nnz=3)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12b3_p.png')
plt.close()


####R^2 TV######
plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==1][:,0],keeprelaxo[keeprelaxo[:,5+n]==1][:,n+8], 'o--', color='red',lw=0.5, ms=2, label='$R^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==1][:,0],keeplasso[keeplasso[:,5+n]==1][:,n+8], 'o--', color='black',lw=0.5, ms=2, label='$R^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$R^2$ (nnz=1)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12c1_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==2][:,0],keeprelaxo[keeprelaxo[:,5+n]==2][:,n+8], 'o--', color='red',lw=0.5, ms=2, label='$R^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==2][:,0],keeplasso[keeplasso[:,5+n]==2][:,n+8], 'o--', color='black',lw=0.5, ms=2, label='$R^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$R^2$ (nnz=2)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12c2_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==3][:,0],keeprelaxo[keeprelaxo[:,5+n]==3][:,n+8], 'o--', color='red',lw=0.5, ms=2, label='$R^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==3][:,0],keeplasso[keeplasso[:,5+n]==3][:,n+8], 'o--', color='black',lw=0.5, ms=2, label='$R^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$R^2$ (nnz=3)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12c3_p.png')
plt.close()



####Adj R^2 TV######
plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==1][:,0],keeprelaxo[keeprelaxo[:,5+n]==1][:,n+10], 'o--', color='red',lw=0.5, ms=2, label='Adj_$R^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==1][:,0],keeplasso[keeplasso[:,5+n]==1][:,n+10], 'o--', color='black',lw=0.5, ms=2, label='Adj_$R^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('Adjusted $R^2$ (nnz=1)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12d1_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==2][:,0],keeprelaxo[keeprelaxo[:,5+n]==2][:,n+10], 'o--', color='red',lw=0.5, ms=2, label='Adj_$R^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==2][:,0],keeplasso[keeplasso[:,5+n]==2][:,n+10], 'o--', color='black',lw=0.5, ms=2, label='Adj_$R^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('Adjusted $R^2$ (nnz=2)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12d2_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==3][:,0],keeprelaxo[keeprelaxo[:,5+n]==3][:,n+10], 'o--', color='red',lw=0.5, ms=2, label='Adj_$R^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==3][:,0],keeplasso[keeplasso[:,5+n]==3][:,n+10], 'o--', color='black',lw=0.5, ms=2, label='Adj_$R^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('Adjusted $R^2$ (nnz=3)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12d3_p.png')
plt.close()


####R^2 Test######
plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==1][:,0],keeprelaxo[keeprelaxo[:,5+n]==1][:,n+9], 'o--', color='red',lw=0.5, ms=2, label='$PR^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==1][:,0],keeplasso[keeplasso[:,5+n]==1][:,n+9], 'o--', color='black',lw=0.5, ms=2, label='$PR^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$PR^2$ (nnz=1)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12e1_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==2][:,0],keeprelaxo[keeprelaxo[:,5+n]==2][:,n+9], 'o--', color='red',lw=0.5, ms=2, label='$PR^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==2][:,0],keeplasso[keeplasso[:,5+n]==2][:,n+9], 'o--', color='black',lw=0.5, ms=2, label='$PR^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$PR^2$ (nnz=2)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12e2_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==3][:,0],keeprelaxo[keeprelaxo[:,5+n]==3][:,n+9], 'o--', color='red',lw=0.5, ms=2, label='$PR^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==3][:,0],keeplasso[keeplasso[:,5+n]==3][:,n+9], 'o--', color='black',lw=0.5, ms=2, label='$PR^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('$PR^2$ (nnz=3)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12e3_p.png')
plt.close()



####Adj R^2 Test######
plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==1][:,0],keeprelaxo[keeprelaxo[:,5+n]==1][:,n+11], 'o--', color='red',lw=0.5, ms=2, label='Adj_$PR^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==1][:,0],keeplasso[keeplasso[:,5+n]==1][:,n+11], 'o--', color='black',lw=0.5, ms=2, label='Adj_$PR^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('Adjusted $PR^2$ (nnz=1)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12f1_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==2][:,0],keeprelaxo[keeprelaxo[:,5+n]==2][:,n+11], 'o--', color='red',lw=0.5, ms=2, label='Adj_$PR^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==2][:,0],keeplasso[keeplasso[:,5+n]==2][:,n+11], 'o--', color='black',lw=0.5, ms=2, label='Adj_$PR^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('Adjusted $PR^2$ (nnz=2)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12f2_p.png')
plt.close()

plt.figure(figsize = (6,4))
plt.plot(keeprelaxo[keeprelaxo[:,5+n]==3][:,0],keeprelaxo[keeprelaxo[:,5+n]==3][:,n+11], 'o--', color='red',lw=0.5, ms=2, label='Adj_$PR^2$_Relaxo')
plt.plot(keeplasso[keeplasso[:,5+n]==3][:,0],keeplasso[keeplasso[:,5+n]==3][:,n+11], 'o--', color='black',lw=0.5, ms=2, label='Adj_$PR^2$_Lasso')
plt.tick_params(axis='both',labelsize=6)  
plt.xlabel('Iteration', fontsize=8)
plt.ylabel('Adjusted $PR^2$ (nnz=3)',fontsize=8)
plt.legend( fontsize = 6)   
plt.savefig(path_data+'img12f3_p.png')
plt.close()



end = time.time() 
print(f"Total runtime of the program is {end - begin}")




##################UAT THE FUNCTION  trainvalid data only##########################

tol=1e-6
model2 = linear_model.Lasso(alpha=1, tol=tol, fit_intercept=False)
model2.fit(trainvalid_data_standardized, trainvalid_y_centered)
print('Coefficients_lasso cek: ', model2.coef_)
pred_train_lasso= model2.predict(trainvalid_data_standardized)
print(mean_squared_error(trainvalid_y_centered,pred_train_lasso))

theta_lasso=coordinate_descent_lasso(initial_theta,trainvalid_data_standardized, trainvalid_y_centered,1, tol)
print(theta_lasso)
print(theta_lasso/std_tv)
y_pred = np.dot(trainvalid_data_standardized,theta_lasso).reshape(-1,1)
mse1=get_MSE(trainvalid_y_centered, y_pred)
print('mse:', mse1)

t=0.5
theta_LSf=theta_LS_full (theta_lasso, trainvalid_data_standardized, trainvalid_y_centered)
theta_relaxo=(t*theta_lasso+(1-t)*theta_LSf)# /std for ori_thetarelaxo_tv
print('theta relaxo:', theta_relaxo)
print('theta relaxo ori:', theta_relaxo/std_tv)




##################UAT THE FUNCTION  on Full Data##########################
print('FULL DATA')
X_standardized, std = normalize_features(X_ori)
y_centered = centered_output(y)
lamda=1
gama=0.5
tol=1e-10

model2 = linear_model.Lasso(alpha=lamda, tol=tol, fit_intercept=False)
model2.fit(X_standardized, y_centered)
print('Coefficients_lasso cek: ', model2.coef_)
print('Coefficients_lasso_ori cek: ', model2.coef_/std)
pred_train_lasso= model2.predict(X_standardized)
print(mean_squared_error(y_centered,pred_train_lasso))

theta_lasso=coordinate_descent_lasso(initial_theta,X_standardized, y_centered,lamda, tol)
print('theta lasso:',theta_lasso)
print('theta lasso ori:',theta_lasso/std)
y_pred = np.dot(X_standardized,theta_lasso).reshape(-1,1)
mse1=get_MSE(y_centered, y_pred)
print('mse:', mse1)


theta_LSf=theta_LS_full (theta_lasso, X_standardized, y_centered)
theta_relaxo=(gama*theta_lasso+(1-gama)*theta_LSf)# /std for ori_thetarelaxo_tv
print('theta relaxo:', theta_relaxo)
print('theta relaxo ori:', theta_relaxo/std)