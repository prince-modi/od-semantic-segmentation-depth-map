import numpy as np
from ultralytics import YOLO
from depth import Depth
import glob,os
import pdb
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from panoptic import PanopticSeg

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

 
def lft(df):
    x1=np.array(df['Mean'])
    x1=x1.reshape(-1,1)
    reg1 = LinearRegression().fit(x1, df['ABS'])
    ypred1=reg1.predict(x1)
    x2=np.array(df['Median'])
    x2=x2.reshape(-1,1)
    reg2 = LinearRegression().fit(x2, df['ABS'])
    ypred2=reg2.predict(x2)
    # print(ypred1,ypred2)
    return ypred1, ypred2


def interpolation(df):
    X_seq = np.linspace(df['Median'].min(),df['Median'].max()).reshape(-1,1)
    coefs = np.polyfit(df['Median'].values.flatten(), df['ABS'].values.flatten(), 3)
    return X_seq, coefs

def polyreg(df):
    degree=2
    x1=np.array(df['Median'])
    x1=x1.reshape(-1,1)
    polyreg_median=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg_median.fit(x1,df['ABS'])
    x2=np.array(df['Mean'])
    x2=x2.reshape(-1,1)
    polyreg_mean=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg_mean.fit(x2,df['ABS'])

    return polyreg_median, polyreg_mean

if __name__=="__main__":
