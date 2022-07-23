# Processing
### Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st

#### Reading the csv file
cancer_data = pd.read_csv("survey lung cancer.csv")

#### Converting text to binary numbers
pd.set_option('mode.chained_assignment', None)
for col in cancer_data.select_dtypes('object'):
    cancer_data[col][cancer_data[col] =='M'] = 0
    cancer_data[col][cancer_data[col] == 'F'] = 1
    cancer_data[col][cancer_data[col] == 'NO'] = 0
    cancer_data[col][cancer_data[col] == 'YES'] = 1

# LOGISTIC REGRESSION (from scratch without librairy)
def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost_function(X,y,w,b):
    m=X.shape[0]
    cost = 0.
    for i in range(m):
        fwb = sigmoid(np.dot(w,X[i])+b)
        loss = -y[i]*np.log(fwb) - (1-y[i])*np.log(1-fwb)
        cost+=loss
    return cost / m

def gradient_calcul(X,y,w,b):
    m,n = X.shape
    djdb = 0.
    djdw = np.zeros((n,)) # for the number of features
    for i in range(m):
        fwb = sigmoid(np.dot(X[i],w)+b)
        diff = fwb-y[i]
        djdb += diff
        for j in range(n):
            djdw[j] += diff * X[i,j]
    return djdw/m,djdb/m


def gradient_descent(X,y,w_in,b_in,alpha,iters):
    m,n = X.shape
    w = w_in
    b=b_in
    print(f"Initial cost : {cost_function(X,y,w,b)}")
    for iteration in range(iters):
        djdw,djdb = gradient_calcul(X,y,w_in,b_in)
        b -= alpha*djdb
        w -= alpha * djdw
    print(f"Final cost : {cost_function(X,y,w,b)}")
    return w,b

X = cancer_data[['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ','ALLERGY ','WHEEZING','ALCOHOL CONSUMING','COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN']]
y = cancer_data['LUNG_CANCER'] 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=42) 
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

#### Cost at each step with the gradient descent  
w_in = np.zeros((X.shape[1]))
b_in = 0.
alpha = 0.003
iters = 20000
#w_final,b_final = gradient_descent(X_train,y_train,w_in,b_in,alpha,iters) # takes some time

#### Final predictions on the test set
def final_prediction(X_test,y_test,w,b):
    m,n = X_test.shape
    p=np.zeros(m)
    print("Test on the test set")
    for i in range(m): # examples
        fwb = sigmoid(np.dot(X_test[i],w)+b)
        if(fwb<0.5): #threshold
            p[i]=0
        else:
            p[i]=1
    return (p==y_test).sum(),(p!=y_test).sum()

#good_preds,bad_preds = final_prediction(X_test,y_test,w_final,b_final)
#print(good_preds)

def spe_pred(X,w,b):
    m = X.shape[0]
    p=0.
    #for i in range(m): # examples
    fwb = sigmoid(np.dot(X,w)+b)
    st.write(f"Your probability of having Lung Cancer is {fwb}")
    if(fwb>0.5): #threshold
        st.write("Therefore we can consider that you potentially have Lung Cancer.")
    else:
        st.write("Therefore we can consider that you don't have Lung Cancer.")

def interface():
    st.title("Lung Cancer prediction using logistic regression from scratch")
    st.info("Before starting, know that 1 corresponds to No and 2 to yes (and 1 for M, 2 for W)")
    st.warning("Note : the model is trained on aged patients, before 52 years old it doesn't predict well")
    #st.write("Wait for the algorithm calculus...")
    w = np.array([-0.24789577,-0.09846541,-0.14310047,0.3016247,0.2128516,0.57180479,0.31619902,0.67509743,1.05731802,0.36177969,0.89612407,0.7023849,-0.15299983,0.79663349,0.03357194])
    b = -0.5628063991598001 # we found optimal w and b values with the previous gradient descent
    X = []
    labels = ["Smoking ?","Yellow fingers ?","Anxiety ?","Peer pressure ? (influenced by a peer)","Chronic disease ?","Fatigue ?","Allergy ?",
    "Wheezing ?","Alcohol consuming ?","Coughing ?","Shortness of breath ?","Swallowing difficulty ?","Chest pain ?"]
    gender = st.number_input("Your gender",step=1,min_value=0,max_value=1)
    age = st.slider("Your age",0,100)
    X.append(gender)
    X.append(age)
    for e in labels:
        X.append(st.number_input(e,step=1,min_value=1,max_value=2))
    X=np.array(X)
    spe_pred(X,w,b)

interface()

# Note : the model is trained on aged patients, before 55 years old it doesn't predict well