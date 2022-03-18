import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, roc_auc_score

class Helper():
    def __init__(self):
        return
 

    def all_to_numeric(self,df):
        '''
        This function transfer all non-numeric data in dataframe to numeric data
        
        Returns
        -------
        df: pandas DataFrame (with numeric values)
        '''
        df = df.copy()
        for col in df.columns:
            se = pd.to_numeric(df[col],errors='coerce')
            df.loc[:,col] = se.to_list()
        return df
 

    def plot_ROC_curve(self, model, X, y_test):
        '''
        Plot single model's ROC curve
        
        Parameters
        ----------
        model: machine learning model or classifier
        X: data input for prediction (ndarrays)
        y_test: true labels or true values of the data (ndarrays)
        '''
        fig = plt.figure()
        Y_predict_prob = model.predict_proba(X)[:,1]
        lr_fpr, lr_tpr, thresholds = roc_curve(y_test, Y_predict_prob)
        # false negative rate = 1 - truth positive rate
        # We assume the loss of every false negative rate is 5 times than falso positive rate
        # We define the loss function: loss = 5*fnr + fpr
        loss = 5*(1-lr_tpr)+lr_fpr
        ix = np.argmin(loss)
        print('Best Threshold=%.4f' % (thresholds[ix]))
        roc_auc = roc_auc_score(y_test, Y_predict_prob)
        plt.title('ROC {}'.format(model))
        plt.plot(lr_fpr, lr_tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1],'r--', label='No Skill')
        plt.scatter(lr_fpr[ix], lr_tpr[ix], marker='o', color='black', label='Best')
        plt.legend(loc = 'lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        
    def plot_ROC_curves(self,model_list,X,y_test):
        '''
        Plot multiple (4) models' ROC curves
        
        Parameters
        ----------
        model_list: machine learning models or classifiers (list)
        X: data input for prediction (ndarrays)
        y_test: true labels or true values of the data (ndarrays)
        '''
        length = len(model_list)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(15,10))
        for model,ax in [(model_list[0],ax1),(model_list[1],ax2),(model_list[2],ax3),(model_list[3],ax4)]:
            Y_predict_prob = model.predict_proba(X)[:,1]
            lr_fpr, lr_tpr, thresholds = roc_curve(y_test, Y_predict_prob)

            roc_auc = roc_auc_score(y_test, Y_predict_prob)
            ax.set_title('ROC {}'.format(model))
            ax.plot(lr_fpr, lr_tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            ax.plot([0, 1], [0, 1],'r--', label='No Skill')
            ax.legend(loc = 'lower right')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_ylabel('True Positive Rate')
            ax.set_xlabel('False Positive Rate')
        fig.tight_layout()

        
    def set_threshold(self,threshold,model,X_test):
        '''
        Parameters
        ----------
        threshold: float number between 0 and 1
        model: machine learning model or classifier
        X_test: data input for prediction (ndarrays)
        
        Returns
        -------
        predicted: List contains int value 1 or 0 
        '''
        predicted_proba = model.predict_proba(X_test)
        predicted = (predicted_proba [:,1] >= threshold).astype('int')
        
        return predicted
