# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

if __name__ == "__main__":
    columns = [0, 5, 8, 9]
    target = 13
    
    # fara filename, scrapy sau beautiful soup
    # imobiliare sau alt site
    filename = "housing.data"
    
    df = pd.read_fwf(filename, header=None)
    
    df_x = df[columns]
    df_y = df[target]
    
    X = df_x.values
    y = df_y.values
    
    A = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    
    y_p = X.dot(A)
    
    # cum evaluam y_p?
    # comparare cu y
    # ascundere parte din date (e.g. 10%) care sa fie
    # folosite pentru evaluare
    # o diferenta dintre y si y_p; diferenta sa fie absoluta