import argparse
import sys
from keras.utils.np_utils import to_categorical
import sklearn
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from root_numpy import root2array, tree2array, array2root
import ROOT
import matplotlib as mpl
#To allow running in batch mode
mpl.use('Agg')
import matplotlib.pyplot as plt
from common_function import dataset, AMS, read_data, prepare_data, calc_sig
#For now working with the NN config file
import config_OPT_NN as conf

def BDTModelada(max_depth, learning_rate, n_estimators, algorithm):
    BDTada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         algorithm=algorithm)
    return BDTada

def BDTModelgrad(max_depth, learning_rate, n_estimators,verbose):#,n_iter_no_change):
    BDTgrad = GradientBoostingClassifier(verbose=verbose,
                                         #n_iter_no_change=n_iter_no_change,
                                         learning_rate=learning_rate,
                                         n_estimators=n_estimators)
    return BDTgrad



"""Boosted Decision Tree Optimisation   

Usage:  
  python Train_BDT.py

Options: 
  -h --help             Show this screen. 
Optional arguments 
  --output =<output>    Specify output name
  --model =<model> Specify signal model ('HVT' or 'GM')
  --depth=<depth> Maximal depth of estimators in Ensemble 
  --nest=<nest> Number of estimators in Ensemble 
  --early=<early> Early stopping for Gradient Boosting Regressor

  --v=<verbose> Set Verbose level
  --lr=<lr> Learning rate for SGD optimizer
  --opt=<opt> Chose between Ensemble Methods AdaBoost and Gradient Boosting Regressor (0 or 1)
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'BDT optimisation')
    parser.add_argument("--v", "--verbose", help="increase output verbosity", default=0, type=int)
    parser.add_argument("--model", help="Specify Model (HVT or GM)", default='GM', type=str)
    parser.add_argument("--output", help="Specify Output name", default='', type=str)
    parser.add_argument('--depth', help = "Specifies the maximal depth of trees in Ensemble", default=3, type=int)
    parser.add_argument('--nest', help = "Specifies the number of estimators in Ensemble", default=400, type=int)
    parser.add_argument('--lr','--learning_rate', help = "Specifies the learning rate", default=0.01, type=float)
    parser.add_argument('--early', help = "Specifies the condition for early stopping for Gradient Boosting Regressor", default=0, type=int)
    parser.add_argument('--opt', help = "Specifies the optimizer used: AdaBoost:0 (default) and Gradient Boost:1,", default=0, type=int)

    args = parser.parse_args()
    print('Train with hyper parameters:')
    print(args)

    #Load input_sample class from config file
    input_sample=conf.input_samples

    #Read data files
    data_set=prepare_data(input_sample,args.model)

    algorithm = "SAMME.R"

    #Define Ensemble models with given hyper parameters
    model_ada=BDTModelada(args.depth, args.lr, args.nest, algorithm)
    model_boost=BDTModelgrad(args.depth, args.lr, args.nest,args.v)#,args.early)

    opt = [model_ada,model_boost]

    #Show number of events in each category
    shape_train=data_set.X_train.shape
    shape_valid=data_set.X_valid.shape
    #shape_test=data_set.X_test.shape

    num_train=shape_train[0]
    num_valid=shape_valid[0]
    #num_test=shape_test[0]
    num_tot=num_train+num_valid#+num_test


    print("The number of training events {0} validation events {1} and total events {2}".format(num_train,num_valid,num_tot))

    #Show model
    print(opt[args.opt])

    print("Fit Model")

    opt[args.opt].fit(data_set.X_train.values, data_set.y_train.values.ravel())


    # Plot the two-class decision scores
    plot_colors = "rb"
    plot_step = 0.025
    twoclass_output = opt[args.opt].decision_function(data_set.X_train.values)
    test_output =  opt[args.opt].decision_function(data_set.X_valid.values)
    plot_range = (twoclass_output.min(), twoclass_output.max())
    plt.subplot(111)
    class_names = "BS"

    print(twoclass_output.shape)
    print(data_set.y_train.values.shape)
    print('Save decision plot')

    for i, n, c in zip(range(2), class_names, plot_colors):
        plt.hist(twoclass_output[data_set.y_train.values[:,0] == i],
                 bins=20,
                 range=plot_range,
                 facecolor=c,
                 label='Class %s' % n,
                 alpha=.5,
                 #edgecolor='k',
                 log=True)
        x1, x2, y1, y2 = plt.axis()

     # Make a colorful backdrop to show the clasification regions in red and blue
    plt.axvspan(0, twoclass_output.max(), color='blue',alpha=0.08)
    plt.axvspan(twoclass_output.min(),0, color='red',alpha=0.08)

    plt.axis((x1, x2, y1, y2 * 1.2))
    plt.legend(loc='upper right')
    plt.ylabel('Counts/Bin')
    plt.xlabel('BDT output')
    #plt.title('Decision Scores')
    plt.title('')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    plt.savefig('./ControlPlots/Decision_score_BDT'+args.output+args.model+'.png')
    plt.clf()

    #Calculate significance in output range between lower and upper
    massindex=0
    mass=300

    lower=40
    upper=70
    step=2

    prob_predict_train_BDT = opt[args.opt].decision_function(data_set.X_train.values)
    prob_predict_valid_BDT = opt[args.opt].decision_function(data_set.X_valid.values)
    
    print(prob_predict_train_BDT)

    highsig = calc_sig(data_set, prob_predict_train_BDT, prob_predict_valid_BDT, lower,upper,step,mass,massindex,'BDT',args.output, args.model)

    print("Save Model")
    filenameBDT = './OutputModel/'+args.model+'_'+str(round(highsig,3))+'_'+args.output+'modelBDT_train.pkl'
    _ = joblib.dump(opt[args.opt], filenameBDT, compress=9)

