# This function estimates the state-space model:
# x_{k+1} =  A x_{k} + B u_{k} + Ke(k)
# y_{k}   =  C x_{k} + e(k)
# Acl= A - KC
     
# Input parameters:
     
# "U" - is the input matrix of the form U \in mathbb{R}^{m \times timeSteps}
# "Y" - is the output matrix of the form Y \in mathbb{R}^{r \times timeSteps}
# "Markov" - matrix of the Markov parameters returned by the function "estimateMarkovParameters()"
# "Z_0_pm1_l" - data matrix returned by the function "estimateMarkovParameters()"      
# "past" is the past horizon
# "future" is the future horizon
# Condition: "future" <= "past"
# "order_estimate" - state order estimate
     
# Output parameters:
# the matrices: A,Acl,B,K,C
# s_singular - singular values of the matrix used to estimate the state-sequence
# X_p_p_l   - estimated state sequence    
     
import glob 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# This function estimates the Markov parameters of the state-space model:
# x_{k+1} =  A x_{k} + B u_{k} + Ke(k)
# y_{k}   =  C x_{k} + e(k)
# The function returns the matrix of the Markov parameters of the model
# Input parameters:
 
# "U" - is the input vector of the form U \in mathbb{R}^{m \times timeSteps}
# "Y" - is the output vector of the form Y \in mathbb{R}^{r \times timeSteps}
# "past" is the past horizon
 
# Output parameters:
#  The problem beeing solved is
#  min_{M_pm1} || Y_p_p_l -  M_pm1 Z_0_pm1_l ||_{F}^{2}
# " M_pm1" - matrix of the Markov parameters
# "Z_0_pm1_l" - data matrix used to estimate the Markov parameters,
# this is an input parameter for the "estimateModel()" function
# "Y_p_p_l" is the right-hand side 
 
 
def estimateMarkovParameters(U,Y,past):
    import numpy as np
    import scipy 
    from scipy import linalg
     
    timeSteps=U.shape[1]
    m=U.shape[0]
    r=Y.shape[0]
    l=timeSteps-past-1
     
    # data matrices for estimating the Markov parameters
    Y_p_p_l=np.zeros(shape=(r,l+1))
    Z_0_pm1_l=np.zeros(shape=((m+r)*past,l+1))  # - returned
    # the estimated matrix that is returned as the output of the function
    M_pm1=np.zeros(shape=(r,(r+m)*past))   # -returned
     
     
    # form the matrices "Y_p_p_l" and "Z_0_pm1_l"
    # iterate through columns
    for j in range(l+1):
        # iterate through rows
        for i in range(past):
            Z_0_pm1_l[i*(m+r):i*(m+r)+m,j]=U[:,i+j]
            Z_0_pm1_l[i*(m+r)+m:i*(m+r)+m+r,j]=Y[:,i+j]
        Y_p_p_l[:,j]=Y[:,j+past]
        # numpy.linalg.lstsq
        #M_pm1=scipy.linalg.lstsq(Z_0_pm1_l.T,Y_p_p_l.T)
        M_pm1=np.matmul(Y_p_p_l,linalg.pinv(Z_0_pm1_l))
     
    return M_pm1, Z_0_pm1_l, Y_p_p_l
###############################################################################
# end of function
     
def estimateModel(U,Y,Markov,Z_0_pm1_l,past,future,order_estimate):
    import numpy as np
    from scipy import linalg
     
    timeSteps=U.shape[1]
    m=U.shape[0]
    r=Y.shape[0]
    l=timeSteps-past-1
    n=order_estimate
     
    Qpm1=np.zeros(shape=(future*r,past*(m+r)))
    for i in range(future):
        Qpm1[i*r:(i+1)*r,i*(m+r):]=Markov[:,:(m+r)*(past-i)]
     
     
     
    # estimate the state sequence
    Qpm1_times_Z_0_pm1_l=np.matmul(Qpm1,Z_0_pm1_l)
    Usvd, s_singular, Vsvd_transpose = np.linalg.svd(Qpm1_times_Z_0_pm1_l, full_matrices=True)
    # estimated state sequence
    X_p_p_l=np.matmul(np.diag(np.sqrt(s_singular[:n])),Vsvd_transpose[:n,:])    
     
     
    X_pp1_pp1_lm1=X_p_p_l[:,1:]
    X_p_p_lm1=X_p_p_l[:,:-1]
     
    # form the matrices Z_p_p_lm1 and Y_p_p_l
    Z_p_p_lm1=np.zeros(shape=(m+r,l))
    Z_p_p_lm1[0:m,0:l]=U[:,past:past+l]
    Z_p_p_lm1[m:m+r,0:l]=Y[:,past:past+l]
     
    Y_p_p_l=np.zeros(shape=(r,l+1))
    Y_p_p_l=Y[:,past:]
         
    S=np.concatenate((X_p_p_lm1,Z_p_p_lm1),axis=0)
    ABK=np.matmul(X_pp1_pp1_lm1,np.linalg.pinv(S))
     
    C=np.matmul(Y_p_p_l,np.linalg.pinv(X_p_p_l))
    Acl=ABK[0:n,0:n]
    B=ABK[0:n,n:n+m]  
    K=ABK[0:n,n+m:n+m+r]  
    A=Acl+np.matmul(K,C)
     
     
    return A,Acl,B,K,C,s_singular,X_p_p_l
##################


csvfilename = 'great-bipedal-bumblebee_all.csv'
df = pd.read_csv(csvfilename)
feature_x = [k for k in df.columns if k not in ['ID', 'System', 't'] and not k.endswith('_n')]
feature_y = ['X_n', 'Y_n']
x_data = df[feature_x].to_numpy()
y_data = df[feature_y].to_numpy()
X = x_data
y = y_data
print(X,y,)



###############################################################################
# model estimation and validation
 
# estimate the Markov parameters
past_value=10 # this is the past window - p 
Markov,Z, Y_p_p_l =estimateMarkovParameters(X,y,past_value)
 
# estimate the system matrices
model_order=3 # this is the model order \hat{n}
Aid,Atilde,Bid,Kid,Cid,s_singular,X_p_p_l = estimateModel(X,y,Markov,Z,past_value,past_value,model_order)  
 
plt.plot(s_singular, 'x',markersize=8)
plt.xlabel('Singular value index')
plt.ylabel('Singular value magnitude')
plt.yscale('log')
#plt.savefig('singular_values.png')
plt.show()
 
# # estimate the initial state of the validation data
# h=10 # window for estimating the initial state
# x0est=estimateInitial(Aid,Bid,Cid,input_val,Y_val,h)
 
# # simulate the open loop model 
# Y_val_prediction,X_val_prediction = systemSimulate(Aid,Bid,Cid,input_val,x0est)
 
# # compute the errors
# relative_error_percentage, vaf_error_percentage, Akaike_error = modelError(Y_val,Y_val_prediction,r,m,30)
# print('Final model relative error %f and VAF value %f' %(relative_error_percentage, vaf_error_percentage))
 
# # plot the prediction and the real output 
# plt.plot(Y_val[0,:100],'k',label='Real output')
# plt.plot(Y_val_prediction[0,:100],'r',label='Prediction')
# plt.legend()
# plt.xlabel('Time steps')
# plt.ylabel('Predicted and real outputs')
# #plt.savefig('results3.png')
# plt.show()
 
#               end of code
###################################################################