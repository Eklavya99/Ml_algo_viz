import imp
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import plots
import numpy as np
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("HyperTUNE")

algos = ['Logistic Regression','KNN', 'Decision tree', 'Random Forest', 'XGB', 'SVM']
data = ['Moons', 'Circles']


col1, col2 = st.columns(2)

algorithm = col1.selectbox('Pick an algorithm', algos)
dataset = col2.selectbox('Pick a dataset', data)


st.sidebar.caption("Dataset's parameters :")
def generate_data(dataset):
        if dataset == data[0]:
                n = st.sidebar.select_slider('Sample size', [100*i for i in range(2, 13, 2)], 400)
                Noise = st.sidebar.select_slider('Noise', [i/10 for i in range(0, 11, 2)], 0.2)
                return make_moons(n_samples = n, noise = Noise, random_state=1) 
        elif dataset == data[1]:
                n = st.sidebar.select_slider('Sample size', [100*i for i in range(2, 13, 2)], 400)
                Noise = st.sidebar.select_slider('Noise', [i/10 for i in range(0, 11, 2)], 0.2)
                return make_circles(n_samples = n, noise = Noise, factor = 0.6 ,random_state=0)
                    
X, y = generate_data(dataset)


st.sidebar.caption("Algorithm's HyperParameters :")
def add_params(algorithm):
        params = {}
        if algorithm == algos[0]:
                maxItr = st.sidebar.select_slider('# iterations', [100, 150, 200])
                regu = st.sidebar.radio('Regularization type', ['l1', 'l2', 'none'], 1)
                lambda_ = st.sidebar.slider('Cost of regularization', 0.0, 1.0, 1.0, 0.001)
                params['maxItr'] = maxItr
                params['lambda'] = lambda_
                params['Regularization Type'] = regu

        elif algorithm == algos[1]:
                n = st.sidebar.number_input('K', 1, 30, value = 5)
                params['K'] = n
                w = st.sidebar.checkbox('Take distance between points into account')
                if w:
                        st.sidebar.write('Closer points are given more weights')
                        params['weight'] = 'distance'
                else:
                        st.sidebar.write('All points are given equal weights')
                        params['weight'] = 'uniform'

        elif algorithm == algos[2]:
                maxDepth = st.sidebar.number_input('Maximum depth', 0, 50, 0)
                minLeaf = st.sidebar.number_input('Minimun leaf size', 1, 50, 1)

                if maxDepth == 0:
                        params['maxDepth'] = None
                else:
                        params['maxDepth'] = maxDepth
                params['minLeaf'] = minLeaf
        
        elif algorithm == algos[3]:
                numTrees = st.sidebar.number_input('Number of trees', 5, 100, 10)
                maxDepth = st.sidebar.number_input('Maximum depth', 0, 50, 0)
                if maxDepth == 0:
                        params['maxDepth'] = None
                else:
                        params['maxDepth'] = maxDepth
                minLeaf = st.sidebar.number_input('Minimun leaf size', 1, 50, 1)

                params['numTrees'] = numTrees
                params['minLeaf'] = minLeaf

        elif algorithm == algos[4]:
                l = st.sidebar.radio('Loss function to optimize over', ['deviance', 'exponential'])
                alpha = st.sidebar.select_slider('Learning rate', [0.1, 0.01, 0.2, 0.02, 0.3, 0.4, 0.5])
                n = st.sidebar.select_slider('N estimators', [i for i in range(10, 101, 10)], 100)
                maxDepth = st.sidebar.number_input('Maximum depth', 3, 51)
                minLeaf = st.sidebar.number_input('Minimum leaf size', 1, 50)

                params['loss'] = l
                params['alpha'] = alpha
                params['n'] = n
                params['maxDepth'] = maxDepth
                params['minLeaf'] = minLeaf
        else:
                kernel = st.sidebar.selectbox('Kernel', ['rbf', 'poly', 'linear', 'sigmoid'])
                params['kernel'] = kernel
                C = st.sidebar.select_slider('C', [0.1, 0.01, 0.001, 0.2, 0.02, 0.3, 0.03, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 1.0)
                params['C'] = C
                if kernel == 'poly':
                        degree = st.sidebar.select_slider('Degree', [i for i in range(2, 11)], 3)
                        params['degree'] = degree
                else:
                        params['degree'] = 3
                if kernel in ['rbf', 'poly', 'sigmoid']:
                        gamma = st.sidebar.radio('Gamma', ['scale', 'auto'])
                        params['gamma'] = gamma
                shrink = st.sidebar.checkbox('Enable shrinking', True)
                if shrink:
                        st.sidebar.write('Shrinking enabled')
                        params['shrink'] = True
                else:
                        st.sidebar.write('Shrinking disabled')
                        params['shrink'] = False

        return params

                
params = add_params(algorithm)
st.write('Current set of hyperParameters - ', params)

def model_LR(params):
        return LogisticRegression(penalty=params['Regularization Type'], max_iter=params['maxItr'], C = params['lambda'], solver = 'saga')
def model_KNN(params):
        return KNeighborsClassifier(n_neighbors = params['K'], weights = params['weight'])
def model_DT(params):
        return DecisionTreeClassifier(min_samples_leaf = params['minLeaf'], max_depth = params['maxDepth'], random_state=42)
def model_RF(params):
        return RandomForestClassifier(n_estimators = params['numTrees'], min_samples_leaf = params['minLeaf'], max_depth = params['maxDepth'])
def model_XGB(params):
        return GradientBoostingClassifier(loss = params['loss'], learning_rate = params['alpha'], max_depth = params['maxDepth'], min_samples_leaf = params['minLeaf'], n_estimators = params['n'])
def model_SVM(params):
        if params['kernel'] != 'linear':
                return SVC(kernel = params['kernel'], C = params['C'], degree = params['degree'], gamma = params['gamma'], shrinking = params['shrink'])
        else:
                return SVC(kernel = params['kernel'], C = params['C'], shrinking = params['shrink'])


X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


def get_model(algorithm, params):
        model = None
        if algorithm == algos[0]:
                model = model_LR(params)
        elif algorithm == algos[1]:
                model = model_KNN(params)
        elif algorithm == algos[2]:
                model = model_DT(params)
        elif algorithm == algos[3]:
                model = model_RF(params)
        elif algorithm == algos[4]:
                model = model_XGB(params)
        else:
                model = model_SVM(params)
        return model

model = get_model(algorithm, params)

model.fit(X_train, y_train)
y_preds = model.predict(X_test)
y_preds_train = model.predict(X_train)

Trainacc = accuracy_score(y_pred = y_preds_train, y_true = y_train)
Testacc = accuracy_score(y_pred = y_preds, y_true = y_test)


ax1 = plots.train_data_plot(X_train, y_train)
ax2 = plots.test_data_plot(X_test, y_test)
ax3 = ax3 = plots.boundry(model, X_train,X_test, y_train, y_test, X)

colxx, colyy = st.columns(2)
with colxx:
        st.write('Train data shape - ', X_train.shape)
        st.metric('Train accuracy' , round(Trainacc,3))
        st.pyplot(ax1)
with colyy:
        st.write('Test data shape - ', X_test.shape)
        st.metric('Test accuracy' , round(Testacc,3))
        st.pyplot(ax2)

with st.container():
       st.pyplot(ax3)




                        
ax4 = plots.roc(lambda x : model.predict_proba(x), X_test, y_test)
ax5 = plots.conf_matx(model, X_test, y_test)
cola, colb = st.columns(2)
with cola:
        st.pyplot(ax4)
with colb:
        st.pyplot(ax5)


    

        
