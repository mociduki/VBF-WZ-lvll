import keras
from keras.utils.np_utils import to_categorical
import ROOT, pickle
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

    debug=False

    if debug: print("s, b=",s,"\t",b)
    
    br = 0.00001 #KM: systematic unc?
    sigma=math.sqrt(b+br)
    n=s+b+br
    radicand = 2 *( n * math.log (n*(b+br+sigma)/(b**2+n*sigma+br))-b**2/sigma*math.log(1+sigma*(n-b)/(b*(b+br+sigma))))

    significance= 0
    if radicand < 0: 
        if debug: print('AMS: radicand is negative. Returning 0.')
    else:
        significance= math.sqrt(radicand)
        pass
    
    return significance

def read_data_apply(filepath, tr_files, Label, variables,model,apply_transform=True,debug=False):
    data = read_data(filepath)

    nFold=len(tr_files)

    X = data[variables]

    if debug: print("before\n",X)
    if apply_transform:
        #X= (X-X_mean)/X_dev
        tr_lists = list()
        for trFile in tr_files: 
            path2read = "OutputModel/"+trFile
            if debug:print ("reading transformation info from: ",path2read)
            tr_lists .append(pickle.load(open( path2read, 'rb' )))

        X_folds = list()
        for findex in range(nFold):
            X_folds.append(X[data['EventNumber']%nFold==findex])
            x_mean,x_dev = tr_lists[findex][0], tr_lists[findex][1]
            X_folds[-1] = (X_folds[-1] - x_mean)/x_dev # transform
            pass

        if debug: 
            for x_f in X_folds: print(np.shape(x_f))            #print(x_f)

        X = pd.concat(X_folds) # unsorted
        X = X.sort_index()     # sort it again ac 2 the original order, to assign masslabel below

        if debug: print(type(X),np.shape(X))

        pass
    if debug: print("after\n",X)

    if (Label>-1): X['LabelMass']=Label
    else:
        if model=='GM':            prob=np.load('probGM.npy')
        elif model=='HVT':         prob=np.load('probHVT.npy')

        label=np.random.choice(prob.shape[0],X.shape[0], p=prob)
        #X['LabelMass'] = label
        X['LabelMass'] = get_mass_label(data["M_WZ"])
        pass

    data = data.reset_index(drop=True) # now remove index again
    return data, X


def read_data(filename):
    root = ROOT.TFile(filename)
    tree = root.Get('nominal')
    cuts='Jet1Pt>0&&Jet2Pt>0&&M_jj>100.&&abs(Weight)<10'
    #KM: now the folding division is applied in the dataset class below
    #if nFold>1: cuts+='&&EventNumber%{0}!={1}'.format(nFold,Findex)
    #print('Applying cuts=',cuts)
    array = tree2array(tree, selection=cuts)
    #print(filename,": nEvents bfr & aft cut=",tree.GetEntries()," ",np.shape(array))
    return pd.DataFrame(array)

class dataset:
    def __init__(self,data,frac_train,variables,model,nFold,Findex,transform,apply_transform=True):
        full=data.sample(frac=1)#,random_state=42)
        #test=data.drop(full.index)

        train = full[(full['EventNumber'])%nFold!=Findex] #x-valid here
        #train=full.sample(frac=frac_train,random_state=42)
        validation=full.drop(train.index)

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

        self.W_train=train[['WeightNormalized']]
        self.W_valid=validation[['WeightNormalized']]
        #self.W_test=test[['Weight']]

        #self.evtNum_train=train[['EventNumber']]
        self.evtNum_valid=validation[['EventNumber']]
        #self.ch_train=train[['isMC']]
        self.ch_valid=validation[['isMC']]

        X_train = train[variables]
        X_valid = validation[variables]

        #Save mean and std dev separately for both models
        if not(model=='GM' or model=='HVT'): raise NameError('Model needs to be either GM or HVT')
        #np.save('./mean'+model, np.mean(X_train))
        #np.save('./std_dev'+model, np.std(X_train))
        transform.append(np.mean(X_train))
        transform.append(np.std(X_train))

        if apply_transform:
            self.X_train= (X_train-np.mean(X_train))/np.std(X_train)
            self.X_valid= (X_valid-np.mean(X_train))/np.std(X_train)
        else:
            self.X_train= X_train
            self.X_valid= X_valid
            pass
        
        #self.X_test= self.X_test-np.mean(self.X_test)
        #self.X_test= self.X_test/np.std(self.X_test)

        self.X_train['LabelMass']=train[['LabelMass']]
        self.X_valid['LabelMass']=validation[['LabelMass']]

        self.mass_train_label=train[['LabelMass']]
        self.mass_valid_label=validation[['LabelMass']]
        #self.X_test['LabelMass']=test[['LabelMass']]

def get_mass_label(mWZ):

    bars = [225,275,325,375,425,475,550,650,750,850,950]

    #mass_labels = np.invert(mWZ<=bars[0]) # smallest range, index=0
    mass_labels = np.zeros(len(mWZ)) # smallest range, index=0

    for i in range(1,len(bars)-1):
        mass_labels += ( ((bars[i] < mWZ) & (mWZ < bars[i+1])) * i )

    mass_labels += (mWZ > bars[-1]) * len(bars)

    return mass_labels
        
def prepare_data(input_samples,model,Findex,nFold,arg_switches=list()):
    #Read background and signal files and save them as panda data frames

    #Names of bck samples
    namesbkg = input_samples.bckgr["name"]
    xsbkg = input_samples.bckgr["xs"]
    neventsbkg = input_samples.bckgr["nevents"]
    #Read files one by one and normalize weights to 150 fb^-1
    bg = None
    print('\nRead Background Samples')
    for i in range(len(namesbkg)):
        sample = read_data(input_samples.filedir+namesbkg[i])
        print(namesbkg[i])
        #sample['Weight']=sample['Weight']*input_samples.lumi*xsbkg[i]/neventsbkg[i] #KM: This is done in WeightNormalized
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
        switches   = input_samples.sigGM["switch" ]
        if len(arg_switches)==len(switches): switches = arg_switches
        namessig   = list() #input_samples.sigGM["name"   ]
        xssig      = list() #input_samples.sigGM["xs"     ]
        neventssig = list() #input_samples.sigGM["nevents"]
        for i in range(len(switches)):
            if not switches[i]: continue
            namessig   .append( input_samples.sigGM["name"   ][i] )
            xssig      .append( input_samples.sigGM["xs"     ][i] )
            neventssig .append( input_samples.sigGM["nevents"][i] )
            pass
        #print(namessig,len(namessig))
    elif model=='HVT':
        switches   = input_samples.sigHVT["switch" ]
        if len(arg_switches)==len(switches): switches = arg_switches
        namessig   = list() #input_samples.sigHVT["name"   ]
        xssig      = list() #input_samples.sigHVT["xs"     ]
        neventssig = list() #input_samples.sigHVT["nevents"]
        for i in range(len(switches)):
            if not switches[i]: continue
            namessig   .append( input_samples.sigHVT["name"   ][i] )
            xssig      .append( input_samples.sigHVT["xs"     ][i] )
            neventssig .append( input_samples.sigHVT["nevents"][i] )
            pass
        #print(namessig,len(namessig))
    else :
        raise NameError('Model needs to be either GM or HVT')

    sig = None
    prob = np.empty(len(namessig))
    print('\nRead Signal Samples')
    for i in range(len(namessig)):
        sample = read_data(input_samples.filedirsig+namessig[i])
        #sample['Weight']=sample['Weight']*input_samples.lumi*xssig[i]/neventssig[i]  #KM: This is done in WeightNormalized
        sample['LabelMass'] = get_mass_label(sample['M_WZ']) #sample['LabelMass'] = i
        print(namessig[i],"\tLabelMass=",i)
        prob[i] = sample.shape[0] 
        if sig is None:
            sig=sample
        else:
            sig=sig.append(sample)#, sort=True)
    #Probability distribution for random Mass Label
    prob=prob/float(sig.shape[0])
    sig['Label'] = '1'

    #Apply random mass label to bkg
    label= get_mass_label(bg['M_WZ']) #np.random.choice(len(namessig),bg.shape[0], p=prob)

    bg['LabelMass'] = label

    #Save prob distribution
    if model=='GM':        np.save('./probGM', prob)
    elif model=='HVT':     np.save('./probHVT', prob)

    #KM: now add sig+bkg to get entire 'data' sample
    data=bg.append(sig)#, sort=True)
    #data.loc[data.m_Valid_jet3 == 0, ['m_Eta_jet3','m_Y_jet3','m_Phi_jet3']] = -10., -10., -5.
    data = data.sample(frac=1,random_state=42).reset_index(drop=True)
    # Pick a random seed for reproducible results
    transform=list()
    data_cont = dataset(data,input_samples.trafrac,input_samples.variables,model,nFold,Findex,transform)
    return data_cont,switches,transform

#Draws Control plot for Neural Network classification
def drawfigure(model,prob_predict_train_NN,data,X_test,nameadd,cut_value,Findex,nFold,sub_dir):
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
    plt.savefig(sub_dir + "/NN_clf_"+nameadd+('_F{0}o{1}'.format(Findex,nFold))+".png")
    plt.clf() 
    return

def calc_sig_new(data_set, prob_predict_train, prob_predict_valid, file_string, sub_dir, masspoints, mass=200, apply_trva_norm=True, apply_mass_window=False, use_abs_weight=False, nbins=1000, debug=False):
    nFold = int(file_string[len(file_string)-1:])
    
    # CONTROL FLAGS             #                                                   #To be consistent with Benjamin
    #apply_mass_window          # apply mass window cut for signif. calculation     #True
    #apply_trva_norm            # training vs validation normalization              #True
    #use_abs_weight             # flip the negative event weights from generator    #True
    do_single_mass   =(mass>0)  # evaluate singificance using one mass point or not #True
    #debug,nbins = True,100

    mass_idx=-1 #default for inclusive masses
    if do_single_mass: mass_idx=masspoints.index(mass)

    if not do_single_mass: apply_mass_window=False

    num_train=len(prob_predict_train)
    num_valid=len(prob_predict_valid)
    num_tot=num_train+num_valid

    #label for sig=0, bkg=1
    label_train = np.reshape(data_set.y_train.values, (len(data_set.y_train.values),))
    label_valid = np.reshape(data_set.y_valid.values, (len(data_set.y_valid.values),))

    #mass label for 200GeV=0, 900GeV=10
    mlabel_train = np.reshape(data_set.mass_train_label.values, (len(data_set.mass_train_label.values),))
    mlabel_valid = np.reshape(data_set.mass_valid_label.values, (len(data_set.mass_valid_label.values),))

    mWZ_train = np.reshape(data_set.mass_train.values, (len(data_set.mass_train.values),))
    mWZ_valid = np.reshape(data_set.mass_valid.values, (len(data_set.mass_valid.values),))

    #sample weight including #generated events & x-section
    weight_train = np.reshape(data_set.W_train.values, (len(data_set.W_train.values),)).copy()
    weight_valid = np.reshape(data_set.W_valid.values, (len(data_set.W_valid.values),)).copy()

    #KM: these two lines below are necessary to normalized the number of events that are significantly different in the training and validation samples
    if apply_trva_norm:
        weight_train *= num_tot/float(num_train)
        weight_valid *= num_tot/float(num_valid)
        pass

    if use_abs_weight:
        weight_train = abs(weight_train)
        weight_valid = abs(weight_valid)
        pass

    if mass_idx<0: print( "Nevents(train), Nevents(valid)= ",len(label_train), len(label_valid))

    indices_tr_s = np.where( label_train=='1' )[0]
    indices_tr_b = np.where( label_train=='0' )[0]
    indices_va_s = np.where( label_valid=='1' )[0]
    indices_va_b = np.where( label_valid=='0' )[0]

    if do_single_mass:
        indices_tr_s = np.where((label_train=='1') & (mlabel_train==mass_idx))[0]
        indices_tr_b = np.where( label_train=='0'                            )[0]
        indices_va_s = np.where((label_valid=='1') & (mlabel_valid==mass_idx))[0]
        indices_va_b = np.where( label_valid=='0'                            )[0]
        if apply_mass_window:
            indices_tr_s = np.where( (label_train=='1') & (mass-mass*0.08*1.5<mWZ_train) & (mWZ_train<mass+mass*0.08*1.5) & (mlabel_train==mass_idx) )[0]
            indices_tr_b = np.where( (label_train=='0') & (mass-mass*0.08*1.5<mWZ_train) & (mWZ_train<mass+mass*0.08*1.5)                            )[0]
            indices_va_s = np.where( (label_valid=='1') & (mass-mass*0.08*1.5<mWZ_valid) & (mWZ_valid<mass+mass*0.08*1.5) & (mlabel_valid==mass_idx) )[0]
            indices_va_b = np.where( (label_valid=='0') & (mass-mass*0.08*1.5<mWZ_valid) & (mWZ_valid<mass+mass*0.08*1.5)                            )[0]
            pass
        pass
    
    p_tr_s = prob_predict_train[indices_tr_s] #sig training sample
    p_tr_b = prob_predict_train[indices_tr_b] #bkg training sample
    p_va_s = prob_predict_valid[indices_va_s] #sig validation sample
    p_va_b = prob_predict_valid[indices_va_b] #bkg validation sample

    w_tr_s = weight_train[indices_tr_s] #sig training sample
    w_tr_b = weight_train[indices_tr_b] #bkg training sample
    w_va_s = weight_valid[indices_va_s] #sig validation sample
    w_va_b = weight_valid[indices_va_b] #bkg validation sample

    #define nPoints for graphing
    graph_points_tr_x=np.zeros(nbins)
    #graph_points_va_x=np.zeros(nbins) # same as above
    graph_points_tr_y=np.zeros(nbins)
    graph_points_va_y=np.zeros(nbins)

    #histograming
    counts_tr_s,bins,_=plt.hist(p_tr_s,bins=nbins,range=(0,1),weights=w_tr_s)  #    print(np.shape(p_tr_s),np.shape(w_tr_s))
    counts_tr_b,   _,_=plt.hist(p_tr_b,bins=nbins,range=(0,1),weights=w_tr_b)  #    print(np.shape(p_tr_b),np.shape(w_tr_b))
    counts_va_s,   _,_=plt.hist(p_va_s,bins=nbins,range=(0,1),weights=w_va_s)  #    print(np.shape(p_va_s),np.shape(w_va_s))
    counts_va_b,   _,_=plt.hist(p_va_b,bins=nbins,range=(0,1),weights=w_va_b)  #    print(np.shape(p_va_b),np.shape(w_va_b))

    print("TRAIN: nbins=",len(counts_tr_s),",\tNsig,Nbkg=",np.sum(counts_tr_s),"\t",np.sum(counts_tr_b), ",\tNsig+Nbkg=",np.sum(counts_tr_s)+np.sum(counts_tr_b))
    print("VALID: nbins=",len(counts_va_s),",\tNsig,Nbkg=",np.sum(counts_va_s),"\t",np.sum(counts_va_b), ",\tNsig+Nbkg=",np.sum(counts_va_s)+np.sum(counts_va_b))

    Nsig_init, sig_yeild, sig_eff, largestAMS, cut_w_maxAMS=0,0,0,0,0
    for i in range(len(counts_tr_s)):
        indices2sum = list(range(i,len(counts_tr_s)))

        Nsig_tr=counts_tr_s[indices2sum].sum()
        Nbkg_tr=counts_tr_b[indices2sum].sum()
        Nsig_va=counts_va_s[indices2sum].sum()
        Nbkg_va=counts_va_b[indices2sum].sum()

        if i==0: Nsig_init=Nsig_va
        #if Nsig_va*nFold<1: continue # if the number of signal events (expected without x-validation) left after cut < 1, skip the rest

        # negative weights to be positive
        #        if Nsig_tr<0:Nsig_tr=abs(Nsig_tr)
        #        if Nbkg_tr<0:Nbkg_tr=abs(Nbkg_tr)
        #        if Nsig_va<0:Nsig_va=abs(Nsig_va)
        #        if Nbkg_va<0:Nbkg_va=abs(Nbkg_va)
        
        significance_tr=0
        significance_va=0

        if Nsig_tr>0 and Nbkg_tr>0:significance_tr= AMS(Nsig_tr*nFold ,Nbkg_tr*nFold)
        if Nsig_va>0 and Nbkg_va>0:significance_va= AMS(Nsig_va*nFold ,Nbkg_va*nFold)

        if debug:
            print("Bin#\t",i)
            print("tr: s, b, sig=", Nsig_tr,"\t", Nbkg_tr,"\t",significance_tr)
            print("va: s, b, sig=", Nsig_va,"\t", Nbkg_va,"\t",significance_va)
            pass
        
        if significance_va>largestAMS: largestAMS, cut_w_maxAMS, sig_yeild, sig_eff= significance_va, bins[i], Nsig_va, Nsig_va/Nsig_init

        graph_points_tr_x[i]=bins[i]
        graph_points_tr_y[i]=significance_tr
        graph_points_va_y[i]=significance_va
        pass

    plt.clf()

    plt.plot(graph_points_tr_x,graph_points_tr_y, label='train')
    plt.plot(graph_points_tr_x,graph_points_va_y, label='valid')
    plt.legend()

    output_file = sub_dir+'/significance_'+file_string+"_m{}".format(mass)+'.png'
    print("Saving sinificance plot: ",output_file)

    plt.savefig(output_file)
    plt.clf()

    print("Signal efficiency with cut for best significance: ",sig_eff, ",\t yeilding {} signal events".format(sig_yeild))

    return largestAMS, cut_w_maxAMS

def calc_sig(data_set,prob_predict_train, prob_predict_valid,lower,upper,step,mass,massindex,file_string):
    #KM: these are arrays of graph points, to be used later in the plotting section
    AMS_train=np.zeros(((upper-lower)//step,2))
    AMS_valid=np.zeros(((upper-lower)//step,2))

    index2=0 #KM: indexing of graph points

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
            if (Yhat_train[index]==1.0 and #predicted to be signal
                data_set.y_train.values[index,0]=='1' and # label to be a signal
                data_set.mass_train.iloc[index,0]>mass-mass*0.08*1.5 and # M_WZ to be between a given range
                data_set.mass_train.iloc[index,0]<mass+mass*0.08*1.5 and # M_WZ to be between a given range 
                data_set.mass_train_label.iloc[index,0]==massindex):
                s_train +=  abs(data_set.W_train.iat[index,0]*(num_tot/float(num_train)))
            elif (Yhat_train[index]==1.0 and
                  data_set.y_train.values[index,0]=='0' and 
                  data_set.mass_train.iloc[index,0]>mass-mass*0.08*1.5 and 
                  data_set.mass_train.iloc[index,0]<mass+mass*0.08*1.5):
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
        avg_ams=ams_valid#(ams_train+ams_valid)/2
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
    plt.savefig('./ControlPlots/significance_'+file_string+'.png')
    plt.clf()
    
    #return AMS_valid[np.argmax(AMS_valid[:,1]),1],cut_w_maxAMS
    return largestAMS,cut_w_maxAMS

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

