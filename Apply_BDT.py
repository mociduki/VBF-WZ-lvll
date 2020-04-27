import argparse
import sklearn
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import math
from root_numpy import root2array, tree2array, array2root
import ROOT
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.models import model_from_json
from common_function import read_data_apply
import config_OPT_NN as conf

def calculate_pred(model,X):
    prob_predict=model.predict_proba(X)
    pcutNN = 0.0
    Yhat=prob_predict[:,0] > pcutNN
    return Yhat, prob_predict

def save_file(data, pred, proba, filename, model):
    data['isSignal'] = pred
    print(filename)
    #for index in range(20):
    #    print "Proba {}".format(proba[index,0])
    data['probSignal'] = proba[:,0]
    array2root(np.array(data.to_records()), 'OutputRoot/new_BDT_'+model+'_'+filename, 'nominal', mode='recreate')
    return

def analyze_data(filedir,filename, model, X_mean, X_dev, label, variables, sigmodel):
    data, X = read_data_apply(filedir+filename, X_mean, X_dev, label, variables, sigmodel)
    pred, proba = calculate_pred(model,X)
    save_file(data, pred, proba, filename, sigmodel)

"""Run Trained BDT on samples
Usage:
  python3 Apply_BDT.py 

Options:
  -h --help             Show this screen.
Optional arguments
  --input =<input>    Specify input name of trained BDT
  --model =<model> Specify signal model ('HVT' or 'GM')
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Apply NN on ntuples')
    parser.add_argument("--input", help="Name of saved trained BDT", default='GM_modelBDT_train.pkl', type=str)
    parser.add_argument("--model", help="Specify Model (HVT or GM)", default='GM', type=str)

    args = parser.parse_args()
    print(args)


    #Load input_sample class from config file
    input_sample=conf.input_samples
    apply_sample=conf.apply_samples

    #Restores Model and compiles automatically
    model = joblib.load('./OutputModel/'+args.input)
    print(model)

    #Load Mean and std dev
    if args.model=='GM':
        X_mean = np.load('meanGM.npy')
        X_dev = np.load('std_devGM.npy')
    elif args.model=='HVT':
        X_mean = np.load('meanHVT.npy')
        X_dev = np.load('std_devHVT.npy')
    else :
        raise NameError('Model needs to be either GM or HVT')

    #Apply NN on all samples in config file
    list_bkg = apply_sample.list_apply_bkg
    if args.model=='GM': 
        list_sig = apply_sample.list_apply_sigGM
    elif args.model=='HVT':
        list_sig = apply_sample.list_apply_sigHVT  
    print('Applying on bkg sample')
    for i in range(len(list_bkg)):
        analyze_data(apply_sample.filedirbkg,list_bkg[i],model, X_mean, X_dev,-1,input_sample.variables,args.model)
    print('Applying on sig sample')
    for i in range(len(list_sig)):
        analyze_data(apply_sample.filedirsig,list_sig[i],model, X_mean, X_dev,i,input_sample.variables,args.model)
