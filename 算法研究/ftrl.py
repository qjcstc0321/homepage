# coding: utf-8

"""
FTRL在线学习算法，相关论文：《Ad Click Prediction- a View from the Trenches》
"""


from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc


class FTRL(object):
    def __init__(self, l1, l2, alpha, beta, n_feature, w=None):
        '''
        Args:
            :param l1: float, L1 regularization parameter
            :param l2: float, L2 regularization parameter
            :param alpha: float, alpha in the per-coordinate rate
            :param beta: float, beta in the per-coordinate rate
            :param n_feature: number of features
            :param w: list or array-like, initialized model weight
        '''
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.beta = beta
        self.n_feature = n_feature
        if w is None:
            self.w = np.zeros(n_feature)
        else:
            if len(w) != n_feature:
                raise ValueError('Different dimension for w and feature!')
            self.w = w

        self.n = np.zeros(n_feature)
        self.z = np.zeros(n_feature)

    def __logloss(self, y, score):
        '''
        Args:
            :param y: int, 0 or 1 truth target
            :param score: float, predict score
        Return:
            logloss: float
        '''
        score = max(min(score, 1.0 - 1e-15), 1e-15)
        logloss = -np.log(score) if y == 1.0 else -np.log(1.0-score)

        return logloss

    def __predict_one(self, x):
        '''
        predict one sample score
        Args:
            :param x: array-like, feature vector
        Return:
            score: float, predict score
        '''
        wTx = np.dot(self.w, x)
        score = 1.0 / (1.0 + np.exp(-wTx))

        return score

    def __update(self, x, y):
        '''
        update model params
        Args:
            :param x: array-like, feature vector
            :param y: int, 0 or 1 truth target
        Return:
            p: float, probability
            logloss: float
        '''
        nonzero_col = np.nonzero(x)[0]      # index of none zero element in feature vector
        p = self.__predict_one(x)
        logloss = self.__logloss(y, p)

        # update z, n
        for i in nonzero_col:
            gi = (p - y) * x[i]
            sigma = (np.sqrt(self.n[i]+gi**2) - np.sqrt(self.n[i])) / self.alpha
            self.z[i] += gi - sigma*self.w[i]
            self.n[i] += gi**2

        # update w
        for j in nonzero_col:
            sgn_z = np.sign(self.z[j])
            if (sgn_z*self.z[j]) <= self.l1:
                self.w[j] = 0.0
            else:
                self.w[j] = (sgn_z*self.l1 - self.z[j]) / ((self.beta+np.sqrt(self.n[j])) / self.alpha + self.l2)

        return p, logloss

    def train(self, df_master, input_vars, target='target', epoch=5):
        '''
        Args:
            :param df_master: DataFrame, training data
            :param input_vars: list or array-like, input variables
            :param target: str, target column name
            :param epoch: int, training epoch
        Return:
            None
        '''
        X = df_master[input_vars].values
        Y = df_master[target].values
        n_row = X.shape[0]      # number of sample
        score = np.empty(n_row)     # result of predict score

        for epoch_n in range(epoch):
            print('\n ====================== Epoch {0}/{1} ====================== \n'.format(epoch_n+1, epoch))
            loss = 0.0
            start_time = datetime.now()
            random_index = np.random.permutation(n_row)  # shuffle index

            # update weight
            for i in random_index:
                p, loss_one = self.__update(X[i], Y[i])
                score[i] = p
                loss += loss_one

            DF = (np.abs(self.w) > 1e-15).sum()     # number of none zero feature
            fpr, tpr, threshold = roc_curve(Y, score)
            train_auc = round(auc(fpr, tpr), 3)     # caculate auc value
            print('time cost: {0}, DF: {1}, logloss: {2}, AUC: {3}'.format(datetime.now()-start_time, DF, loss, train_auc))

    def predict(self, df_master, input_vars):
        '''
        Args
            :param df_master: DataFrame, data for predict
            :param input_vars: list or array-like, input variables
        Return:
            score: array-like, predict score
        '''
        X = df_master[input_vars].values
        wTX = np.dot(X, self.w)
        score = 1.0 / (1.0 + np.exp(-wTX))

        return score

    def save_model(self, input_vars, save_path):
        '''
        save model
        Args
            :param input_vars: list or array-like, input variables
            :param save_path: string, model save path
        Return:
            score: array-like, predict score
        '''
        df_tmp = pd.DataFrame()
        df_tmp['Var_Name'] = input_vars
        df_tmp['Coefficient'] = self.w
        df_tmp.to_csv(save_path, index=False)
