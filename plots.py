from cProfile import label
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


def train_data_plot(X_train, y_train):
        fig = plt.figure(figsize=(7, 7))
        plt.title('Training data')
        fig.patch.set_facecolor('white')
        plt.scatter(X_train[:, 0], X_train[:, 1], alpha = 0.8, c = y_train, s = 3**3, marker = 'x', cmap = 'PRGn')
        return fig


def test_data_plot(X_test, y_test):
        fig = plt.figure(figsize=(7,7))
        plt.title('Test data')
        plt.scatter(X_test[:, 0], X_test[:, 1], alpha = 0.8, c = y_test, s = 3**3, marker = '+', cmap =  'bwr')

        return fig

def boundry(model, X_train, X_test, y_train, y_test, X):
        ax = plt.figure(figsize=(10, 10))
        x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
        y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
        h = 0.7
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0009", "#0000FF"])
        if hasattr(model, 'decision_function'):
                Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
                Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        plt.title('Decision Boundry')
        plt.contourf(xx, yy, Z, cmap = cm, alpha = 0.8)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train , marker = '^', cmap=cm_bright, edgecolors="k", label = 'Train data')
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker = 'v',cmap=cm_bright, alpha=0.8, edgecolors="k", label = 'Test data')
        plt.xlabel('x1')
        plt.ylabel('x2')
        return ax

def roc(model, X_test, y_test):
        decision_test = model(X_test)
        if len(decision_test.shape) == 2:
                decision_test = decision_test[:, 1]
        fpr, tpr, th = roc_curve(y_test, decision_test)
        auc = roc_auc_score(y_true=y_test, y_score=decision_test)

        fig = plt.figure(figsize=(4, 4 ))
        plt.title('ROC curve', fontsize = 9)
        plt.text(x = 0.55, y = 0.0, s = f"AUC score = {round(auc, 2)}")
        plt.plot(fpr, tpr, color = '#5751f0')
        plt.xlabel('False positive rate', fontsize=8)
        plt.ylabel('True positive rate', fontsize = 8)
        return fig

def conf_matx(model, X_test, y_test):
        ax = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = 'Blues', values_format = ' .3g')
        plt.title('Confusion Matrix')
        plt.show()
        


# def plot_decision_boundary(model, X, y):
#     # Set min and max values and give it some padding
#     fig = plt.figure(figsize=(10, 10))
#     x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
#     y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
#     h = 0.2
#     # Generate a grid of points with distance h between them
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     # Predict the function value for the whole grid
#     Z = model(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     # Plot the contour and training examples
#     plt.contourf(xx, yy, Z, cmap='RdBu')
#     plt.ylabel('x2')
#     plt.xlabel('x1')
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
#     return fig
