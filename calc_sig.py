import ROOT
from ROOT import gROOT, gDirectory, gPad
from ROOT import TFile
from ROOT import TTree
from ROOT import TH1F
from ROOT import TPad
from ROOT import TCanvas
from ROOT import TObject
from ROOT import TDirectory
import config_OPT_NN as conf
from common_function import AMS

variable = "pSignal"
cut=""

h_ref = TH1F("ref","ref",50,0,1);

def get_sig_hist():

    h_sig=h_ref.Clone("h_sig")
    for idx in range(len(conf.input_samples.sigGM['name'])):
        # read file and access tree
        new_name = "OutputRoot/new_GM_"+conf.input_samples.sigGM['name'][idx]
        print("Accessing file:",new_name)
        file=TFile.Open(new_name)
        tree=file.Get("nominal")

        # prepare hist
        h_tmp= h_sig.Clone("h_tmp{0}".format(idx))
        print(tree.GetName(),tree.GetEntries(),h_tmp.GetName())
        
        # project tree to a hist
        tree.Project(h_tmp.GetName(),variable,cut)
        h_tmp.Scale(conf.input_samples.lumi*conf.input_samples.sigGM['xs'][idx]/conf.input_samples.sigGM['nevents'][idx]);

        h_sig.Add(h_tmp)
        pass
    return h_sig

def get_bkg_hist():

    h_bkg=h_ref.Clone("h_bkg")#TH1F("bkg","bkg",500,0,1)
    for idx in range(len(conf.input_samples.bckgr['name'])):
        # read file and access tree
        new_name = "OutputRoot/new_GM_"+conf.input_samples.bckgr['name'][idx]
        print("Accessing file:",new_name)
        file=TFile.Open(new_name)
        tree=file.Get("nominal")
        
        # prepare hist
        h_tmp= h_bkg.Clone("h_tmp{0}".format(idx))
        
        print(tree.GetName(),tree.GetEntries(),h_tmp.GetName())
        
        # project tree to a hist
        tree.Project(h_tmp.GetName(),variable,cut)
        h_tmp.Scale(conf.input_samples.lumi*conf.input_samples.bckgr['xs'][idx]/conf.input_samples.bckgr['nevents'][idx]);
        
        h_bkg.Add(h_tmp)
        pass
    return h_bkg

def plot_dist(h_sig,h_bkg):
    c1=TCanvas("dist","title",800,600)
#     h_bkg.Rebin(10)
#     h_sig.Rebin(10)
    
    h_bkg.SetLineColor(ROOT.kRed)
    h_sig.SetLineColor(ROOT.kBlue)
    
    h_bkg.Draw("")
    h_sig.Draw("same")
    gPad.SetLogy()
    
    c1.SaveAs("dist.png")
    return

def get_ams(h_sig,h_bkg):

    h_ams=h_ref.Clone("ams");
    for bindex in range(h_sig.GetNbinsX()):
        if bindex==0:continue
        nSig=h_sig.Integral(bindex,h_sig.GetNbinsX())
        nBkg=h_bkg.Integral(bindex,h_bkg.GetNbinsX())

        h_ams.SetBinContent(bindex,AMS(nSig,nBkg))
        #h_ams.SetBinError(bindex,0)

        pass
    return h_ams

def plot_ams(h_ams):
    c1=TCanvas("ams","title",800,600)
    #h_ams.Rebin(10)
    h_ams.Draw("hist ")
    c1.SaveAs("ams.png")
    return
    
h_sig = get_sig_hist()
h_bkg = get_bkg_hist()
h_ams = get_ams(h_sig,h_bkg)

plot_dist(h_sig,h_bkg)
plot_ams(h_ams)
