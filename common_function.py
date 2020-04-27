import keras
from keras.utils.np_utils import to_categorical
import ROOT
from root_numpy import root2array, tree2array, array2root
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from keras import backend as K

def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 0.00001, b = background, s = signal, log is natural logarithm with added systematics"""
    
    br = 0.00001
    sigma=math.sqrt(b+br)
    n=s+b+br
    radicand = 2 *( n * math.log (n*(b+br+sigma)/(b**2+n*sigma+br))-b**2/sigma*math.log(1+sigma*(n-b)/(b*(b+br+sigma))))
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return math.sqrt(radicand)

def read_data_apply(filepath, X_mean, X_dev, Label, variables,model,Findex=0,nFold=1):
    data = read_data(filepath,Findex,nFold)
    data = data.reset_index(drop=True)

    X = data[variables]

    X= X-X_mean
    X= X/X_dev
    if (Label>-1):
        X['LabelMass']=Label
    else:
        if model=='GM':
            prob=np.load('probGM.npy')
        elif model=='HVT':
            prob=np.load('probHVT.npy')
        label=np.random.choice(prob.shape[0],X.shape[0], p=prob)
        X['LabelMass'] = label

    return data, X


def read_data(filename,Findex,nFold):
    root = ROOT.TFile(filename)
    tree = root.Get('nominal')
    cuts='Jet1Pt>0&&Jet2Pt>0&&M_jj>100.'
    if nFold>1: cuts+='&&EventNumber%{0}!={1}'.format(nFold,Findex)
    #print('Applying cuts=',cuts)
    array = tree2array(tree, selection=cuts)
    return pd.DataFrame(array)

class dataset:
    def __init__(self,data,frac_train,frac_valid,variables,model):
        train_full=data.sample(frac=frac_train,random_state=42)
        #test=data.drop(train_full.index)
        train=train_full.sample(frac=frac_valid,random_state=42)
        validation=train_full.drop(train.index)

        #Separate variables from labels
        self.y_train=train[['Label']]#to_categorical(train[['Label']])
        self.y_valid=validation[['Label']]#to_categorical(validation[['Label']])
        #self.y_test=to_categorical(test[['Label']])

        mass_train=train[['M_WZ']]
        mass_valid=validation[['M_WZ']]
        #mass_test=test[['Mass']]

        self.mass_train=mass_train.reset_index(drop=True)
        self.mass_valid=mass_valid.reset_index(drop=True)
        #self.mass_test=mass_test.reset_index(drop=True)

        self.W_train=train[['Weight']]
        self.W_valid=validation[['Weight']]
        #self.W_test=test[['Weight']]

        X_train = train[variables]
        X_valid = validation[variables]

        #Save mean and std dev separately for both models
        if model=='GM':
            np.save('./meanGM', np.mean(X_train))
            np.save('./std_devGM', np.std(X_train))
        elif model=='HVT':
            np.save('./meanHVT', np.mean(X_train))      
            np.save('./std_devHVT', np.std(X_train))
        else :
            raise NameError('Model needs to be either GM or HVT')

        self.X_train= X_train-np.mean(X_train)
        self.X_train= X_train/np.std(X_train)

        self.X_valid= X_valid-np.mean(X_valid)
        self.X_valid= X_valid/np.std(X_valid)
        
        #self.X_test= self.X_test-np.mean(self.X_test)
        #self.X_test= self.X_test/np.std(self.X_test)

        self.X_train['LabelMass']=train[['LabelMass']]
        self.X_valid['LabelMass']=validation[['LabelMass']]

        self.mass_train_label=train[['LabelMass']]
        self.mass_valid_label=validation[['LabelMass']]
        #self.X_test['LabelMass']=test[['LabelMass']]
        
def prepare_data(input_samples,model,Findex,nFold):
    #Read background and signal files and save them as panda data frames

    #Names of bck samples
    namesbkg = input_samples.bckgr["name"]
    xsbkg = input_samples.bckgr["xs"]
    neventsbkg = input_samples.bckgr["nevents"]
    #Read files one by one and normalize weights to 150 fb^-1
    bg = None
    print('Read Background Samples')
    for i in range(len(namesbkg)):
        sample = read_data(input_samples.filedir+namesbkg[i],Findex,nFold)
        print(namesbkg[i])
        sample['Weight']=sample['Weight']*input_samples.lumi*xsbkg[i]/neventsbkg[i]
        if bg is None:
            bg=sample
        else:
            #Sort not working?
            bg=bg.append(sample)#, sort=True)

    #Add label 0 for bkg
    bg['Label'] = '0'

    #Read signal
    #Either GM or HVT model
    if model=='GM':
        namessig = input_samples.sigGM["name"]
        xssig = input_samples.sigGM["xs"]
        neventssig = input_samples.sigGM["nevents"]
    elif model=='HVT':
        namessig = input_samples.sigHVT["name"]
        xssig = input_samples.sigHVT["xs"]
        neventssig = input_samples.sigHVT["nevents"]
    else :
        raise NameError('Model needs to be either GM or HVT')
    sig = None
    prob = np.empty(len(namessig))
    print('Read Signal Samples')
    for i in range(len(namessig)):
        sample = read_data(input_samples.filedirsig+namessig[i],Findex,nFold)
        print(namessig[i])
        sample['Weight']=sample['Weight']*input_samples.lumi*xssig[i]/neventssig[i]
        sample['LabelMass'] = i
        prob[i] = sample.shape[0] 
        if sig is None:
            sig=sample
        else:
            sig=sig.append(sample)#, sort=True)
    #Probability distribution for random Mass Label
    prob=prob/float(sig.shape[0])
    sig['Label'] = '1'

    #Apply random mass label to bkg
    label=np.random.choice(len(namessig),bg.shape[0], p=prob)

    bg['LabelMass'] = label

    #Save prob distribution
    if model=='GM':
        np.save('./probGM', prob)
    elif model=='HVT':
        np.save('./probHVT', prob)

    data=bg.append(sig)#, sort=True)
    #data.loc[data.m_Valid_jet3 == 0, ['m_Eta_jet3','m_Y_jet3','m_Phi_jet3']] = -10., -10., -5.
    data = data.sample(frac=1,random_state=42).reset_index(drop=True)
    # Pick a random seed for reproducible results
    # Use 30% of the training sample for validation

    data_cont = dataset(data,1.,input_samples.valfrac,input_samples.variables,model)
    return data_cont

#Draws Control plot for Neural Network classification
def drawfigure(model,prob_predict_train_NN,data,X_test,nameadd,cut_value,Findex,nFold):
    #pcutNN = np.percentile(prob_predict_train_NN,pcut)

    Classifier_training_S = model.predict(data.X_train.values[data.y_train.values[:,0]=='1'], verbose=False)
    Classifier_training_B = model.predict(data.X_train.values[data.y_train.values[:,0]=='0'], verbose=False)
    Classifier_testing_A = model.predict(X_test, verbose=False)

    c_max = max([max(Classifier_training_S),max(Classifier_training_B),max(Classifier_testing_A)])
    c_min = min([min(Classifier_training_S),min(Classifier_training_B),min(Classifier_testing_A)])
  
    # Get histograms of the classifiers NN
    Histo_training_S = np.histogram(Classifier_training_S,bins=50,range=(c_min,c_max))
    Histo_training_B = np.histogram(Classifier_training_B,bins=50,range=(c_min,c_max))
    Histo_testing_A = np.histogram(Classifier_testing_A,bins=50,range=(c_min,c_max))

    # Lets get the min/max of the Histograms
    AllHistos= [Histo_training_S,Histo_training_B]
    h_max = max([histo[0].max() for histo in AllHistos])*1.2

    # h_min = max([histo[0].min() for histo in AllHistos])
    h_min = 1.0
  
    # Get the histogram properties (binning, widths, centers)
    bin_edges = Histo_training_S[1]
    bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    # To make error bar plots for the data, take the Poisson uncertainty sqrt(N)
    ErrorBar_testing_A = np.sqrt(Histo_testing_A[0])

    # Draw objects
    ax1 = plt.subplot(111)
  
    # Draw solid histograms for the training data
    ax1.bar(bin_centers-bin_widths/2.,Histo_training_B[0],facecolor='red',linewidth=0,
            width=bin_widths,label='B (Train)',alpha=0.5)
    ax1.bar(bin_centers-bin_widths/2.,Histo_training_S[0],bottom=Histo_training_B[0],
            facecolor='blue',linewidth=0,width=bin_widths,label='S (Train)',alpha=0.5)
 
    ff = 1.0*(sum(Histo_training_S[0])+sum(Histo_training_B[0]))/(1.0*sum(Histo_testing_A[0]))
 
     # # Draw error-bar histograms for the testing data
    ax1.errorbar(bin_centers-bin_widths/2, ff*Histo_testing_A[0], yerr=ff*ErrorBar_testing_A, xerr=None, 
                 ecolor='black',c='black',fmt='.',label='Test (reweighted)')
  
    # Make a colorful backdrop to show the clasification regions in red and blue
    ax1.axvspan(cut_value, c_max, color='blue',alpha=0.08)
    ax1.axvspan(c_min,cut_value, color='red',alpha=0.08)
  
    # Adjust the axis boundaries (just cosmetic)
    ax1.axis([c_min, c_max, h_min, h_max])
  
    # Make labels and title
    plt.title("")
    plt.xlabel("Probability Output (NN)")
    plt.ylabel("Counts/Bin")
    plt.yscale('log', nonposy='clip')
 
    # Make legend with smalll font
    legend = ax1.legend(loc='upper center', shadow=True,ncol=2)
    for alabel in legend.get_texts():
        alabel.set_fontsize('small')
        pass
  
    # Save the result to png
    plt.savefig("./ControlPlots/NN_clf_"+nameadd+('_F{0}o{1}'.format(Findex,nFold))+".png")
    plt.clf() 


def calc_sig(data_set,prob_predict_train, prob_predict_valid,lower,upper,step,mass,massindex,mod,name,model,Findex,nFold):
    AMS_train=np.zeros(((upper-lower)//step,2))
    AMS_valid=np.zeros(((upper-lower)//step,2))

    index2=0

    shape_train=data_set.X_train.shape
    shape_valid=data_set.X_valid.shape
    num_train=shape_train[0]
    num_valid=shape_valid[0]
    num_tot=num_train+num_valid#+num_test

    largestAMS,cut_w_maxAMS=0,0
    for cut in range(lower,upper, step):
        print("")
        print("With upper percentile {}".format(cut))
        pcutNN = np.percentile(prob_predict_train,cut)
        #km: this percentile is calculated wrt the entire sample, not the signal efficiency

        print("Cut Value {}".format(pcutNN))
        Yhat_train = prob_predict_train[:] > pcutNN
        Yhat_valid = prob_predict_valid[:] > pcutNN
    
        s_train=b_train=0
        s_valid=b_valid=0

        for index in range(len(Yhat_train)):
            if (Yhat_train[index]==1.0 and data_set.y_train.values[index,0]=='1' and data_set.mass_train.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_train.iloc[index,0]<mass+mass*0.08*1.5 and data_set.mass_train_label.iloc[index,0]==massindex):
                s_train +=  abs(data_set.W_train.iat[index,0]*(num_tot/float(num_train)))
            elif (Yhat_train[index]==1.0 and data_set.y_train.values[index,0]=='0' and data_set.mass_train.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_train.iloc[index,0]<mass+mass*0.08*1.5):
                b_train +=  abs(data_set.W_train.iat[index,0]*(num_tot/float(num_train)))

        for index in range(len(Yhat_valid)):
            if (Yhat_valid[index]==1.0 and data_set.y_valid.values[index,0]=='1' and data_set.mass_valid.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_valid.iloc[index,0]<mass+mass*0.08*1.5 and data_set.mass_valid_label.iloc[index,0]==massindex):
                s_valid +=  abs(data_set.W_valid.iat[index,0]*(num_tot/float(num_valid)))
            elif (Yhat_valid[index]==1.0 and data_set.y_valid.values[index,0]=='0' and data_set.mass_valid.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_valid.iloc[index,0]<mass+mass*0.08*1.5):
                b_valid +=  abs(data_set.W_valid.iat[index,0]*(num_tot/float(num_valid)))

        print("S and B NN training  : S=",s_train,"\tB=",b_train)
        print("S and B NN validation: S=",s_valid,"\tB=",b_valid)

        ams_train=AMS(s_train,b_train)
        ams_valid=AMS(s_valid,b_valid)
        avg_ams=(ams_train+ams_valid)/2
        if avg_ams>largestAMS: largestAMS, cut_w_maxAMS = avg_ams, pcutNN

        print('Calculating AMS score for NNs with a probability cutoff pcut=',cut)
        print('   - AMS based on training   sample:',ams_train)
        print('   - AMS based on validation sample:',ams_valid)
        
        AMS_train[index2,0]=pcutNN
        AMS_train[index2,1]=AMS(s_train,b_train)
        AMS_valid[index2,0]=pcutNN
        AMS_valid[index2,1]=AMS(s_valid,b_valid)
        index2+=1

    plt.plot(AMS_train[:,0],AMS_train[:,1], label='train')
    plt.plot(AMS_valid[:,0],AMS_valid[:,1], label='valid')
    plt.legend()
    #plt.title('Significance as a function of the probability output')
    plt.title('')
    plt.xlabel("Cut value")
    plt.xlim(0,1)
    plt.ylabel("Significance ($\sigma$)")
    plt.savefig('./ControlPlots/significance_'+model+'_'+str(mod)+str(name)+('_F{0}o{1}'.format(Findex,nFold))+'.png')
    plt.clf()
    
    return AMS_valid[np.argmax(AMS_valid[:,1]),1],cut_w_maxAMS

#Atlernative metric to accuracy
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

