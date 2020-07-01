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
    string sdir="0630";
    string model="GM";

    string rootdir = "OutputRoot/" + sdir + '/';
    string savedir = "ControlPlots/" + sdir + "/pSignal/"; // MAKE SURE PSIGNAL EXISTS IN SUBDIRECTORY

    int mass_arr[11];
    float xs[11];
    int nevents[11];
    float filtEff[11];
    int mass_rnd[11];
    int init_ID;
   
    if (model == "GM") {
        mass_arr[0] = 200; mass_arr[1] = 300; mass_arr[2] = 400; mass_arr[3] = 500; mass_arr[4] = 600; mass_arr[5] = 700; mass_arr[6] = 800; mass_arr[7] = 900;
        xs[0] = 7.0596; xs[1] = 3.9238; xs[2] = 2.4428; xs[3] = 1.6113; xs[4] = 1.1005; xs[5] = 0.77398; xs[6] = 0.55433; xs[7] = 0.40394;
        nevents[0] = 70000; nevents[1] = 70000; nevents[2] = 70000; nevents[3] = 70000; nevents[4] = 70000; nevents[5] = 70000; nevents[6] = 70000; nevents[7] = 50000;
        for (int m=0; m<8; m++) { filtEff[m] = 1; }
        init_ID  = 28;
    }

    if (model == "HVT") {
        mass_arr[0] = 300; mass_arr[1] = 400; mass_arr[2] = 500; mass_arr[3] = 600; mass_arr[4] = 700; mass_arr[5] = 800; mass_arr[6] = 900; mass_arr[7] = 1000;
        xs[0] = 10.54; xs[1] = 0.7975; xs[2] = 0.1663; xs[3] = 0.087984; xs[4] = 0.049882; xs[5] = 0.02961; xs[6] = 1.3050; xs[7] = 4.4843;
        nevents[0] = 190000; nevents[1] = 190000; nevents[2] = 190000; nevents[3] = 190000; nevents[4] = 175000; nevents[5] = 190000; nevents[6] = 160000; nevents[7] = 160000;
        for (int m=0; m<8; m++) { filtEff[m] = 1; }
        init_ID = 28;
    }

    // Initial mass and mass file ID
    int mass_num = init_ID;
    
    // Looping on all masses
    for (int i=0; i<8; i++) {

        int mass = mass_arr[i];

        // Calculating signal weight
        float weight = xs[i]*filtEff[i]*140./nevents[i];        

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
        TCanvas *c1 = new TCanvas("c1","c2",800,400);
        c1->Divide(2,1);
        c1->cd(1);
        data->SetLineColor(99);
        data->Draw("pSignal >> pSig",Form("%f",weight),"HIST");
        TH1F *hist = (TH1F*)gDirectory->Get("pSig");
        hist->SetTitle(Form("pSignal - mass %i",mass));
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

        b1->SetLineColor(77);
        b2->SetLineColor(4);
        b1->Draw("pSignal >> pSig_b1","1704*140/3890000.","SAME HIST");
        b2->Draw("pSignal >> pSig_b2","47*140/7325000.","SAME HIST");

        TH1F *hist_b1 = (TH1F*)gDirectory->Get("pSig_b1");
        TH1F *hist_b2 = (TH1F*)gDirectory->Get("pSig_b2");

        // Initializing cut values and integrals
        float cut_value[25];
        float signal_integral[25];
        float bckgd1_integral[25];
        float bckgd2_integral[25];

        for (int j=0; j<21; j++) {
            // Finding the bin associated with the cut value
            cut_value[j] = j/20.;
            int cut_bin  = hist->FindBin(cut_value[j]);
            int high_bin = hist->FindBin(1);

            // Calculating integrals
            signal_integral[j] = hist   ->Integral(cut_bin,high_bin);
            bckgd1_integral[j] = hist_b1->Integral(cut_bin,high_bin);
            bckgd2_integral[j] = hist_b2->Integral(cut_bin,high_bin);
        }

        // Drawing the results
        c1->cd(2);
        TGraph* sig_integral = new TGraph(20, cut_value, signal_integral);
        sig_integral->SetName("sig_integral");
        sig_integral->SetLineWidth(2);
        sig_integral->SetLineColor(99);
        sig_integral->Draw();
        sig_integral->SetTitle("Integrals");

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
	
        // Saving the figure as .png and .root
        string sfname1 = savedir + "pSig_integrals_m" + to_string(mass) + "_and_bckgrd.png";
        string sfname2 = savedir + "pSig_integrals_m" + to_string(mass) + "_and_bckgrd.root";
        char const *sfname1_c = sfname1.c_str();
        char const *sfname2_c = sfname2.c_str();
    	c1->SaveAs(sfname1_c);
    	c1->SaveAs(sfname2_c);
        
        c1->Close();
        
        // Updating the mass and mass file ID
        mass_num += 1;

    	cout << "--------------------------" << endl;
    }
    return 0;
}
