#include <TString.h>
#include <TFile.h>
#include <TMath.h>
#include <TTree.h>
#include <TColor.h>
#include <TH1F.h>
#include <TROOT.h>
#include <TChain.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <iostream>
#include <fstream>
#include <TLine.h>
#include <string>
#include <algorithm>
#include <list>
#include <iostream>
#include <stdio.h>
#include <sys/stat.h>
using namespace std;

//------------------------------------------------------------------------------
/*
Analysis of the pSignal of different masses and backgrounds following the
training of the neuron network at each masses seperatly. 

The pSignal and the normalized number of events after certain 
cut values are plotted. 
*/
//------------------------------------------------------------------------------

int main() {

    // PARAMETERS TO EDIT
    string sdir  = "0630";
    string model = "GM";

    string rootdir = "OutputRoot/" + sdir + '/';
    string savedir = "ControlPlots/" + sdir + "/pSignal/"; // MAKE SURE PSIGNAL EXISTS IN SUBDIRECTORY

    // Initial mass and mass file ID
    int mass_num = 28;
    int mass;  
    if (model == "GM") {
        mass = 200;
    }

    if (model == "HVT") {
        mass = 300;
    }
    
    // Looping on all masses
    for (int i=0; i<9; i++) {
        
        // Creating the path for the data file
        string fdir  = rootdir + "m" + to_string(mass);
        string fname = "new_" + model + "_mainMVA.3050" + to_string(mass_num) + "_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_" + to_string(mass) + "_qcd0_ntuples.root";
        string fpath = fdir + "/" + fname;
        char const *fpath_c = fpath.c_str();

        cout << "Reading signal file : " << fpath_c << endl;
        
        // Reading the signal file
        TFile *sfile = new TFile(fpath_c, "READ");
        TTree *data;
        sfile->GetObject("nominal", data);
        
        // Drawing the data
        TCanvas *c1 = new TCanvas("c1","c2",800,600);
        c1->Divide(2,2);
        c1->cd(1);
        data->SetLineColor(99);
        data->Draw("pSignal >> pSig","WeightNormalized","HIST");
        TH1F *hist = (TH1F*)gDirectory->Get("pSig");
        hist->SetTitle(Form("pSignal - mass %i",mass));
        gPad->SetLogy();

       // Applying condtions to signal
        c1->cd(3);
        data->SetLineColor(99);
        data->Draw("pSignal >> pSig_f","WeightNormalized*(M_jj>500)*(Deta_jj>3.5)","HIST");
        TH1F *hist_f = (TH1F*)gDirectory->Get("pSig_f");
        hist_f->SetTitle("M_jj>500 GeV and Deta>3.5");
        gPad->SetLogy();

        // Reading the background files
        string bname1 = "new_" + model + "_mainMVA.361292_MGaMcAtNloPy8EG_NNPDF30LO_A14NNPDF23LO_WZ_lvll_FxFx_ntuples.root";
        string bname2 = "new_" + model + "_mainMVA.364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_ntuples.root";
    
        string bpath1 = fdir + "/" + bname1;
        string bpath2 = fdir + "/" + bname2;

        char const *bpath1_c = bpath1.c_str();
        char const *bpath2_c = bpath2.c_str();

        cout << "Reading background file 1 : " << bpath1_c << endl;
        cout << "Reading background file 2 : " << bpath2_c << endl;

        TFile *bfile1 = new TFile(bpath1_c, "READ");
        TFile *bfile2 = new TFile(bpath2_c, "READ");
        
        TTree *b1;
        TTree *b2;

        bfile1->GetObject("nominal", b1);
        bfile2->GetObject("nominal", b2);

        c1->cd(1);
        b1->SetLineColor(77);
        b2->SetLineColor(4);
        b1->Draw("pSignal >> pSig_b1","WeightNormalized","SAME HIST");
        b2->Draw("pSignal >> pSig_b2","WeightNormalized","SAME HIST");

        TH1F *hist_b1 = (TH1F*)gDirectory->Get("pSig_b1");
        TH1F *hist_b2 = (TH1F*)gDirectory->Get("pSig_b2");

        // Applying conditions to background
        c1->cd(3);
        b1->SetLineColor(77);
        b2->SetLineColor(4);
        b1->Draw("pSignal >> pSig_b1_f","WeightNormalized*(M_jj>500)*(Deta_jj>3.5)","SAME HIST");
        b2->Draw("pSignal >> pSig_b2_f","WeightNormalized*(M_jj>500)*(Deta_jj>3.5)","SAME HIST");

        TH1F *hist_b1_f = (TH1F*)gDirectory->Get("pSig_b1_f");
        TH1F *hist_b2_f = (TH1F*)gDirectory->Get("pSig_b2_f");

        // Initializing cut values and integrals
        float cut_value[25];
        float signal_integral[25];
        float bckgd1_integral[25];
        float bckgd2_integral[25];
        float signal_integral_f[25];
        float bckgd1_integral_f[25];
        float bckgd2_integral_f[25];

        for (int j=0; j<21; j++) {
            // Finding the bin associated with the cut value
            cut_value[j] = j/20.;
            int cut_bin  = hist->FindBin(cut_value[j]);
            int high_bin = hist->FindBin(1);

            // Calculating integrals
            signal_integral[j] = hist   ->Integral(cut_bin,high_bin);
            bckgd1_integral[j] = hist_b1->Integral(cut_bin,high_bin);
            bckgd2_integral[j] = hist_b2->Integral(cut_bin,high_bin);
            signal_integral_f[j] = hist_f   ->Integral(cut_bin,high_bin);
            bckgd1_integral_f[j] = hist_b1_f->Integral(cut_bin,high_bin);
            bckgd2_integral_f[j] = hist_b2_f->Integral(cut_bin,high_bin);
        }

        // Drawing the results
        c1->cd(2);
        TGraph* sig_integral = new TGraph(20, cut_value, signal_integral);
        sig_integral->SetName("sig_integral");
        sig_integral->SetLineWidth(2);
        sig_integral->SetLineColor(99);
        sig_integral->SetTitle("Integrals; Cut value");
        sig_integral->Draw();

        TGraph* b1_integral = new TGraph(20, cut_value, bckgd1_integral);
        b1_integral->SetName("b1_integral");
        b1_integral->SetLineWidth(2);
        b1_integral->SetLineColor(77);
        b1_integral->Draw("SAME");

        TGraph* b2_integral = new TGraph(20, cut_value, bckgd2_integral);
        b2_integral->SetName("b2_integral");
        b2_integral->SetLineWidth(2);
        b2_integral->SetLineColor(4);
        b2_integral->Draw("SAME");

        auto legend = new TLegend(0.12,0.12,0.32,0.25);
        legend->AddEntry("sig_integral", "signal" ,"lep");
        legend->AddEntry("b1_integral", "QCD background" ,"lep");
        legend->AddEntry("b2_integral", "EW background" ,"lep");
        legend->Draw();

        gPad->SetLogy();

        c1->cd(4);
        TGraph* sig_integral_f = new TGraph(20, cut_value, signal_integral_f);
        sig_integral_f->SetName("sig_integral_f");
        sig_integral_f->SetLineWidth(2);
        sig_integral_f->SetLineColor(99);
        sig_integral_f->SetTitle("Integrals; Cut value");
        sig_integral_f->Draw();

        TGraph* b1_integral_f = new TGraph(20, cut_value, bckgd1_integral_f);
        b1_integral_f->SetName("b1_integral_f");
        b1_integral_f->SetLineWidth(2);
        b1_integral_f->SetLineColor(77);
        b1_integral_f->Draw("SAME");

        TGraph* b2_integral_f = new TGraph(20, cut_value, bckgd2_integral_f);
        b2_integral_f->SetName("b2_integral_f");
        b2_integral_f->SetLineWidth(2);
        b2_integral_f->SetLineColor(4);
        b2_integral_f->Draw("SAME");

        gPad->SetLogy();

        // Saving the figure as .png and .root
        string sfname1 = savedir + "pSig_integrals_m" + to_string(mass) + "_and_bckgrd.png";
        string sfname2 = savedir + "pSig_integrals_m" + to_string(mass) + "_and_bckgrd.root";
        char const *sfname1_c = sfname1.c_str();
        char const *sfname2_c = sfname2.c_str();
    	c1->SaveAs(sfname1_c);
    	c1->SaveAs(sfname2_c);
        
        c1->Close();
        
        // Updating mass and mass file ID
        mass_num += 1;
        mass     += 100;

    	cout << "--------------------------" << endl;
    }
    return 0;
}

void pSignal_cv_plotting() {main();}
