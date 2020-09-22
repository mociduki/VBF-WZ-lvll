import argparse
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras import optimizers
import numpy as np
import pandas as pd
import math, os
from root_numpy import root2array, tree2array, array2root
import ROOT
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.models import model_from_json
from common_function import read_data_apply, calc_sig, f1, f1_loss
import config_OPT_NN as conf
from pathlib import Path

def read_phys_model(file_name):

    s_model_name='GM'
    if file_name.find('_HVT')!=-1: s_model_name='HVT'

    return s_model_name

def read_phys_models(model_files):
    idx=0
    phys_model=''
    for File in model_files:
        if idx==0: phys_model=read_phys_model(File)
        else: 
            if phys_model!=read_phys_model(File): 
                print ("Something is wrong with signal model names, aborting...")
                exit(1)
                pass
            pass
        idx+=1
        pass

    return phys_model

def read_cut_fold(files):

    cut_values=list()
    for File in files:
        cut_values.append(read_cut(File))
        print(cut_values[-1])
        pass

    return cut_values

def read_cut(file_name):

    s_cut=file_name[file_name.find("CV")+2:file_name.find("_F")]
    #print(s_cut)
    
    cut_value=float(s_cut[1:])
    if s_cut.find("m")==0: cut_value*=-1
    #print(cut_value)

    return cut_value

def calculate_pred(model,X,cut_value,verbose):
    if verbose: print('\t nEvents / file=\t',len(X))
    prob_predict=model.predict(X.values, verbose=False)
    #pcutNN = np.percentile(prob_predict,40.)
    Yhat=prob_predict[:] > cut_value
    return Yhat, prob_predict

def calculate_pred_fold(models,data,X,cut_values):

    probabilities=list()
    predictions  =list()

    for idx in range(len(models)):
        pred_fold, prob_fold = calculate_pred(models[idx],X,cut_values[idx],verbose=(idx==0))
        predictions.append(pred_fold)
        probabilities.append(prob_fold)
        pass

    #print('=============================')
    #print(probabilities[0])
    #print('=============================')
    #print(probabilities[1])

    #for i in range(len(probabilities[0])):
    #    x_str = np.array_repr(X.values[i]).replace('\n', '')
    #    print(data["EventNumber"][i],x_str,probabilities[0][i][0])
            
    for idx in range(len(probabilities[0])):
        list_idx=data['EventNumber'][idx]%len(models)#idx%len(models)
        if list_idx!=0:
            probabilities[0][idx] = probabilities[list_idx][idx]
            predictions  [0][idx] = predictions  [list_idx][idx]
            pass
        pass

    #print('=============================')
    #print(probabilities[0])

    pred_fold=predictions  [0]
    prob_fold=probabilities[0]

    #print(len(pred_fold),len(prob_fold))
    
    return pred_fold,prob_fold

def save_file(data, pred, proba, filename, phys_model, sub_dir, syst_var):
    #data['isSignal'] = pred
    data['pSignal_'+phys_model] = proba[:]
    #print("Input file  =\t\t\t",filename)

    # Checking for or creating subdirectory
    sub_dir_or = "OutputRoot/"+sub_dir
    Path(sub_dir_or).mkdir(parents=True, exist_ok=True)

    outputPath=sub_dir_or+'/new_'+filename     #print(outputPath)

    save_mode='update'
    if syst_var=='nominal': save_mode='recreate'

    array2root(np.array(data.to_records(index=False)), outputPath, syst_var, mode=save_mode)

    print('Save file as= {}'.format(outputPath))
    print()

    return

#def analyze_data(filedir,filename, model, X_mean, X_dev, label, variables, sigmodel,cut_value,sub_dir):
#    data, X = read_data_apply(filedir+filename, X_mean, X_dev, label, variables, sigmodel)
#    if len(X)==0: return
#    pred, proba = calculate_pred(model,X,cut_value)
#    save_file(data, pred, proba, filename, sigmodel, sub_dir)

def analyze_data_folds(filedir,filename, models, tr_files, label, variables, sigmodel,cut_values,sub_dir,syst_var,debug=False):
    data, X = read_data_apply(filedir+filename, tr_files, label, variables, sigmodel,syst_var)

    if len(X)==0: return
    #print(len(data),len(X))
    
    pred_fold, proba_fold = calculate_pred_fold(models,data,X,cut_values)
    save_file(data, pred_fold, proba_fold, filename, sigmodel, sub_dir,syst_var)
    if debug:
        for i in range(len(data['EventNumber'])):
            print (data['EventNumber'][i], proba_fold[i][0])
            #x_str = np.array_repr(X.values[i]).replace('\n', '')
            #print (data['EventNumber'][i], x_str, proba_fold[i][0])

"""Run Trained Neural Network on samples
Usage:
  python3 Apply_NN.py 

Options:
  -h --help             Show this screen.
Optional arguments
  --input =<input>    Specify input name of trained NN
  --phys_model =<phys_model> Specify signal phys_model ('HVT' or 'GM')
"""

def parse_model_files(input):

    files=list()
    Findices=list()
    nFold=int(0)
    lockFold=False
    for File in input.split(","):
        files.append(File)
        if '_F' in File:
            f_str=File[File.find('_F')+2:File.rfind("_")]
            
            # sanity check
            if not lockFold:
                nFold =int(f_str[f_str.find("o")+1:])
                lockFold=True
                pass
            Findex=int(f_str[:f_str.find("o")])
            print(File, 'Findex / nFold=\t',Findex,"/",nFold)
            
            if int(f_str[f_str.find("o")+1:])!=nFold:
                print("Something is wrong with number of fold configuration, aborting...")
                exit(1)
                pass

            if not Findex in Findices:
                Findices.append(Findex)
            else:
                print("Same fold index has been found, abroting...")
                exit(1)
                pass

            pass
        pass

    tr_files=list()
    for filename in files:
        tr_file=filename
        tr_file=tr_file.replace(".h5",".pkl")
        tr_files.append(tr_file)
        pass

    return files,tr_files

def read_models(model_files):

    models=list()

    for File in model_files:
        model = load_model('OutputModel/'+File)
        model.summary()
        models.append(model)
        pass

    return models

def read_syst_variations(file):
    syst_variations=list()
    for line in open(args.syst_list, 'r').read().splitlines(): 
        if '#' in line: continue
        syst_variations.append(line)
        pass
    return syst_variations

def read_sample_dir(input_dirpath):
    list_bkg=list() #    input_dirpath=args.target_dir
    for r,d,f in os.walk(input_dirpath): #args.target_dir):
        for file_name in f:
            if not('.root' in file_name): continue
            list_bkg.append(file_name)
            pass
        pass
    return list_bkg

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Apply NN on ntuples')
    parser.add_argument("--input", help="Name of saved trained NN", default='GM_output_NN.h5', type=str)
    parser.add_argument("--sdir", help="Subdirectory of saved output", default="", type=str)
    #parser.add_argument("--syst_var", help="Target systematic variation", default="nominal", type=str)
    parser.add_argument("--syst_list", help="list of systematic variation", default="syst.list", type=str)
    parser.add_argument("--run_systematics", help="Bool whether to run syst variations", default=False, type=bool)
    parser.add_argument("--target_dir", help="Target directory of root files to apply NN ouput", default="", type=str)
    parser.add_argument("--single_file", help="Single target file", default="", type=str)
    # parser.add_argument("--phys_model", help="Specify Model (HVT or GM)", default='GM', type=str)

    args = parser.parse_args()
    print(args)

    model_files,tr_files=parse_model_files(args.input)

    #Load input_sample class from config file
    input_sample=conf.input_samples
    apply_sample=conf.apply_samples

    #Restores Model and compiles automatically
    models = read_models(model_files)

    cut_values= read_cut_fold(model_files)

    phys_model=read_phys_models(model_files)

    #Load Mean and std dev
    if not(phys_model=='GM' or phys_model=='HVT'):
        raise NameError('Model needs to be either GM or HVT')
#    X_mean = np.load('mean'+phys_model+'.npy')
#    X_dev = np.load('std_dev'+phys_model+'.npy')
    #Mean and std dev from training
    #print(X_mean)
    #print(X_dev)

    #list_data = apply_sample.list_apply_data

    #Apply NN on all samples in config file
    list_bkg = apply_sample.list_apply_bkg # not only background but all sample is listed in this

    input_dirpath=apply_sample.filedirapp
    if args.target_dir!='': 
        input_dirpath = args.target_dir
        list_bkg = read_sample_dir(input_dirpath)
        pass
    
    if args.single_file!='':
        list_bkg.clear()
        list_bkg.append(args.single_file)
        pass

    syst_variations=['nominal']
    if args.run_systematics and args.syst_list!="" and not('data' in  args.single_file): syst_variations = read_syst_variations(args.syst_list)
    print(syst_variations)
    
    print('Applying on all samples')
    for bkg_file in list_bkg:
        #print(bkg_file)
        for syst_var in syst_variations:
            #if ("JET" in syst_var) or ("MET" in syst_var) or ("EG" in syst_var) or ("EL" in syst_var): continue
            if ('450772' in bkg_file) and (('FT_EFF_Eigen_C' in syst_var) or syst_var=='FT_EFF_Eigen_Light_0__1down'): continue
            if ('410155' in bkg_file) and syst_var=='MUON_EFF_TTVA_SYS__1up' : continue
            print('Current file and systematic variation= ',bkg_file,"\t",syst_var,end='')
            analyze_data_folds(input_dirpath,bkg_file,models, tr_files,-1,input_sample.variables,phys_model,cut_values,args.sdir,syst_var)
        pass

#    print('Applying on sig sample')
#    for i in range(len(list_sig)):
#        analyze_data_folds(apply_sample.filedirsig,list_sig[i],models, X_mean, X_dev, i,input_sample.variables,phys_model,cut_values)
#        pass

    # print('Applying on data sample')
    # for i in range(len(list_data)):
    #     print('input file  =',list_data[i])
    #     analyze_data(apply_sample.filedirdata,list_data[i],model, X_mean, X_dev,i,input_sample.variables,phys_model,cut_value)
