import argparse
import sys
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
#import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from root_numpy import root2array, tree2array, array2root
from common_function import dataset, AMS, read_data, prepare_data, drawfigure, calc_sig, calc_sig_new, f1, f1_loss
import config_OPT_NN as conf
import ROOT 
from pathlib import Path
import os

def KerasModel(input_dim,numlayer,numn, bool_drop, dropout):
    model = Sequential()
    model.add(Dense(numn, input_dim=int(input_dim)))
    model.add(Activation('relu'))
    if bool_drop: model.add(Dropout(dropout))
    for i in range(numlayer-1):
        model.add(Dense(numn))
        model.add(Activation('relu'))
        if bool_drop: model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

"""Neural Network Optimisation
Usage:
  python3 OPT_VBS_NN.py 

Options:
  -h --help             Show this screen.
Optional arguments
  --v = <verbose> Set verbose level
  --model =<model> Specify signal model ('HVT' or 'GM')
  --output =<output>    Specify output name
  --numlayer=<numlayer>     Specify number of hidden layers.
  --numn=<numn> Number of neurons per hidden layer
  --dropout=<dropout> Dropout to reduce overfitting: 0 to turn off
  --epoch=<epochs> Specify training epochs
  --patience=<patience> Set patience for early stopping
  --lr=<lr> Learning rate for SGD optimizer
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'NN optimisation')
    parser.add_argument("--v", "--verbose", help="increase output verbosity", default=0, type=int)
    parser.add_argument("--model", help="Specify Model (HVT or GM)", default='GM', type=str)
    parser.add_argument("--output", help="Specify Output name", default='', type=str)
    parser.add_argument('--numlayer', help = "Specifies the number of layers of the Neural Network", default=3, type=int)
    parser.add_argument('--numn', help = "Specifies the number of neurons per hidden layer", default=200, type=int)
    parser.add_argument('--lr','--learning_rate', help = "Specifies the learning rate for SGD optimizer", default=0.01, type=float)
    parser.add_argument('--dropout', help = "Specifies the applied dropout", default=0.05, type=float)
    parser.add_argument('--epochs', help = "Specifies the number of epochs", default=80, type=int)
    parser.add_argument('--patience', help = "Specifies the patience for early stopping", default=5, type=int)
    parser.add_argument('--no_train', help = "Skip training process", default=False, type=bool)
    parser.add_argument('--mass_points', help = "mass points to be included in the training", default=list(), type=int,nargs='+')
    parser.add_argument('--nFold', help = "number of folds", default=2, type=int)
    parser.add_argument('--Findex', help = "index in nfolds", default=0, type=int)
    parser.add_argument('--sdir', help = 'Name of subdirectory within Controlplots/', default='', type=str)

    args = parser.parse_args()
    print('\n=====================================================================')
    print('  MODEL       : {}'.format(args.model))
    print('  MASS POINTS : {}'.format(args.mass_points))
    print('  FOLD INDEX  : {}/{}'.format(args.Findex,args.nFold))
    print('=====================================================================\n')

    # Checking for or creating subdirectory
    sub_dir_cp = "ControlPlots/"+args.sdir
    Path(sub_dir_cp).mkdir(parents=True, exist_ok=True)

    #Load input_sample class from config file
    input_sample=conf.input_samples

    #Additional name from model and hyper parameters
    nameadd=args.output+args.model

    mass_list = [200,250,300,350,400,450,500,600,700,800,900]
    if args.model=="HVT": mass_list = [250,300,350,400,450,500,600,700,800,900,1000]

    if args.model=="HVT" and 200 in args.mass_points:
        print ("WARNING: you specified 200 GeV mass point for HVT which does not exist, 200 GeV will be removed!!")
        args.mass_points.remove(200)
    elif args.model=="GM" and 1000 in args.mass_points:
        print ("WARNING: you specified 1000 GeV mass point for GM which does not exist, 1000 GeV will be removed!!")
        args.mass_points.remove(1000)
        pass

    tmp_switches=list() #switches for mass points
    if len(args.mass_points)>0:
        for mass in mass_list: tmp_switches.append(mass in args.mass_points)
        print("INFO: Overriding the mass switches because the mass_points argument was given!")
    else: print("INFO: No mass_points argument was given. Using default mass switches in the config_OP_NN.py!")

    #Read data files
    data_set,tmp_switches=prepare_data(input_sample,args.model,args.Findex,args.nFold,arg_switches=tmp_switches)

    ar_switches = np.array(tmp_switches)
    ar_mass     = np.array(mass_list)
    tmp_array = ar_switches*ar_mass
    args.mass_points = tmp_array.tolist()
    #Remove extra zeros
    while 0 in args.mass_points: args.mass_points.remove(0)
    #print(args.mass_points)

    print("\nUsing mass switches:",end='')
    for a,b in zip(mass_list,tmp_switches): print("({},{}) ".format(a,b),end='')
    print()

    #Get input dimensions
    shape_train=data_set.X_train.shape
    shape_valid=data_set.X_valid.shape
    #shape_test=data_set.X_test.shape

    num_train=shape_train[0]
    num_valid=shape_valid[0]
    #num_test=shape_test[0]

    num_tot=num_train+num_valid#+num_test

    #print(data_set.X_train[(data_set.X_train['LabelMass']==0)])
    
    print('Number of training: {}, validation: {} and total events: {}.'.format(num_train,num_valid,num_tot))

    #Define model with given parameters
    model=KerasModel(shape_train[1],args.numlayer,args.numn,(args.dropout>0),args.dropout)
    
    #Possible optimizers
    sgd = optimizers.SGD(lr=args.lr, decay=1e-6, momentum=0.6, nesterov=True)
    ada= optimizers.Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.01)
    nadam=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model.compile(loss='binary_crossentropy'#f1_loss
                  , optimizer=sgd, metrics=['accuracy'])#,f1])
    model.summary()
    #model.save("modelNN_initial_"+args.model+".h5")

    # Save the model architecture
    #with open('OutputModel/model_architecture.json', 'w') as f:
    #    f.write(model.to_json())

    #Define checkpoint to save best performing NN and early stopping
    path='./OutputModel/'+nameadd+'output_NN.h5'
    #checkpoint=keras.callbacks.ModelCheckpoint(filepath='output_NN.h5', monitor='val_acc', verbose=args.v, save_best_only=True)
    callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience),ModelCheckpoint(filepath=path, monitor='val_loss', verbose=args.v, save_best_only=True)]

    if not args.no_train:
        # Train Model
        logs = model.fit(data_set.X_train.values, data_set.y_train.values, epochs=args.epochs,
                         validation_data=(data_set.X_valid.values, data_set.y_valid.values),batch_size=256, callbacks=callbacks, verbose =args.v, class_weight = 'auto')
        
        # accuracy vs epochs
        plt.plot(logs.history['acc'], label='train')
        plt.plot(logs.history['val_acc'], label='valid')
        plt.legend()
        plt.title('')
        if len(args.mass_points)>0: plt.title("mass points : {}".format(args.mass_points))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(sub_dir_cp + '/class_'+nameadd+('_F{0}o{1}'.format(args.Findex,args.nFold))+'.png')
        plt.clf()
        
        # plot loss vs epochs
        plt.plot(logs.history['loss'], label='train')
        plt.plot(logs.history['val_loss'], label='valid')
        plt.legend() #plt.title('Training loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title('')
        if len(args.mass_points)>0: plt.title("mass points : {}".format(args.mass_points))
        plt.savefig(sub_dir_cp + '/loss_'+nameadd+('_F{0}o{1}'.format(args.Findex,args.nFold))+'.png')
        plt.clf()
        pass

    #Calculate Significance
    #Load saved weights which gave best result on training
    model = load_model(path)#, custom_objects={'f1': f1, 'f1_loss' : f1_loss})

    prob_predict_train_NN = model.predict(data_set.X_train.values, verbose=False)
    prob_predict_valid_NN = model.predict(data_set.X_valid.values, verbose=False)
    #prob_predict_test_NN = model.predict(data_set.X_test, verbose=False)

    #Calculate significance in output
    #highsig,cut_value = calc_sig_new(data_set, prob_predict_train_NN[:,0], prob_predict_valid_NN[:,0], mass=200, '{0}_NN{1}_F{2}o{3}'.format(args.model,args.output,args.Findex,args.nFold) )
    highsig,cut_value = 0,0
    if len(args.mass_points)>1: args.mass_points.append(-1)
    for mass in reversed(args.mass_points):
        print ("\nEvaluating significance curve at mass: {}".format(mass))
        highsig,cut_value = calc_sig_new(data_set, prob_predict_train_NN[:,0], prob_predict_valid_NN[:,0], '{0}_NN{1}_F{2}o{3}'.format(args.model,args.output,args.Findex,args.nFold),sub_dir_cp,args.mass_points,mass=mass)
    
    # Draw figures
    drawfigure(model,prob_predict_train_NN,data_set,data_set.X_valid.values,nameadd,cut_value,args.Findex,args.nFold,sub_dir_cp)

    cv_str=''
    if   cut_value<0: cv_str="m%.3f" % cut_value
    else:             cv_str="p%.3f" % cut_value

    print(cv_str)

    outputName='sigvalid_'+args.model+'_m{}'.format(args.mass_points if len(args.mass_points)==1 else "Multi")+'_S'+str(round(highsig,3))+args.output+"_CV"+cv_str+('_F{0}o{1}'.format(args.Findex,args.nFold))+'_NN'

    #Save model in OutputModel with the highest significance obtained on validation set
    model.save('./OutputModel/'+outputName+'.h5')
