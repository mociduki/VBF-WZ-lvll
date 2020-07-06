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

// SUBDIRECTORIES TO EDIT
string idir  = "0630/";
string tmass = "m900";
string sdir  = idir+tmass;

string get_file_name(int mass) {
  
  string insert_str="main";
  string              file_path="OutputRoot/"+sdir+"/new_GM_"+insert_str+"MVA.305028_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_200_qcd0_ntuples.root";
  if      (mass==300) file_path="OutputRoot/"+sdir+"/new_GM_"+insert_str+"MVA.305029_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_300_qcd0_ntuples.root";
  else if (mass==400) file_path="OutputRoot/"+sdir+"/new_GM_"+insert_str+"MVA.305030_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_400_qcd0_ntuples.root";
  else if (mass==500) file_path="OutputRoot/"+sdir+"/new_GM_"+insert_str+"MVA.305031_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_500_qcd0_ntuples.root";
  else if (mass==600) file_path="OutputRoot/"+sdir+"/new_GM_"+insert_str+"MVA.305032_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_600_qcd0_ntuples.root";
  else if (mass==700) file_path="OutputRoot/"+sdir+"/new_GM_"+insert_str+"MVA.305033_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_700_qcd0_ntuples.root";
  else if (mass==800) file_path="OutputRoot/"+sdir+"/new_GM_"+insert_str+"MVA.305034_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_800_qcd0_ntuples.root";
  else if (mass==900) file_path="OutputRoot/"+sdir+"/new_GM_"+insert_str+"MVA.305035_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_900_qcd0_ntuples.root";

  return file_path;
  
}

int get_color(int mass) {

  int color = kBlack;
  if      (mass==200) color=kGray+2;
  else if (mass==300) color=kMagenta;
  else if (mass==400) color=kBlue; 
  else if (mass==500) color=kCyan+2;
  else if (mass==600) color=kGreen+2; 
  else if (mass==700) color=kYellow+2;
  else if (mass==800) color=kOrange;
  else if (mass==900) color=kRed;

  return color;
  
}

// index           = 0
// Yields          = 0
// isMC            = 0
// Channel         = 0
// Year            = 0
// NormSF          = 0
// WeightNormalized = 0
// Weight          = 0
// PtReweight      = 0

TString title, proj_str;
int nbins = 50; float xmin =0, xmax = 1;

TH1F* get_bkg_hist() {

  TChain* chain = new TChain("nominal");
  TString ins_str="main";
  //chain->Add("OutputRoot/new_GM_"+ins_str+"MVA.364253_Sherpa_222_NNPDF30NNLO_lllv_ntuples.root");
  chain->Add("OutputRoot/"+sdir+"/new_GM_"+ins_str+"MVA.361292_MGaMcAtNloPy8EG_NNPDF30LO_A14NNPDF23LO_WZ_lvll_FxFx_ntuples.root");
  chain->Add("OutputRoot/"+sdir+"/new_GM_"+ins_str+"MVA.364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_ntuples.root");

  TH1F* hist = new TH1F("bkg",title,nbins,xmin,xmax);
  chain->Project(hist->GetName(),proj_str,"","norm");

  return hist;
}

TH1F* get_hist(int mass) {

  TH1F* hist;
  if (mass>0) {
    string fname=get_file_name(mass);
    
    TFile* f = TFile::Open(fname.data(),"read");
    TTree* t = (TTree*)f->Get("nominal");
    
    TString histName = "mass"+TString::Itoa(mass,10);
    //std::cout<<histName<<std::endl;

    hist = new TH1F(histName ,title,nbins,xmin,xmax);
    t->Project(hist->GetName(),proj_str,"","norm");
  }
  else hist = get_bkg_hist();

  hist->SetMaximum(hist->GetBinContent( hist->GetMaximumBin() )*1.8);
  hist->SetLineWidth(2);
  hist->SetLineColor(get_color(mass));
  
  return hist;
}

void nn_per_mass(TString varname="pSignal") {

  if      (varname == "pSignal"     ) title="NN output : "+tmass, proj_str=varname, nbins = 50, xmin =0, xmax = 1;
  else if (varname == "M_WZ"        ) title=varname, proj_str=varname, nbins = 50, xmin =0, xmax = 1500;
  else if (varname == "M_jj"        ) title=varname, proj_str=varname, nbins = 50, xmin =0, xmax = 1500;
  else if (varname == "ZetaLep"     ) title=varname, proj_str=varname, nbins = 50, xmin =-3.5, xmax = 3.5;
  else if (varname == "DY_jj"       ) title=varname, proj_str=varname, nbins = 50, xmin =0, xmax = 10;
  else if (varname == "Deta_jj"     ) title=varname, proj_str=varname, nbins = 50, xmin =0, xmax = 10;
  else if (varname == "Dphi_jj"     ) title=varname, proj_str=varname, nbins = 50, xmin =0, xmax = TMath::Pi();
  else if (varname == "Meff"        ) title=varname, proj_str= "Pt_W+Pt_Z+Jet1Pt+Jet2Pt+Met", nbins = 50, xmin =0, xmax = 2000;
  else if (varname == "dEta_WZ"     ) title=varname, proj_str= "abs(Eta_Z-Eta_W)", nbins = 50, xmin =0, xmax = 5;
  else if (varname == "Eta_Z"       ) title=varname, proj_str= varname, nbins = 50, xmin =-3, xmax = 3;
  else if (varname == "Eta_W"       ) title=varname, proj_str= varname, nbins = 50, xmin =-3, xmax = 3;
  else if (varname == "Pt_Z"        ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 1000;
  else if (varname == "Pt_W"        ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 1000;
  else if (varname == "Met"         ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 500;
  else if (varname == "Njets"       ) title=varname, proj_str= varname, nbins = 5 , xmin =2, xmax = 7;
  else if (varname == "NBjets"      ) title=varname, proj_str= varname, nbins = 5 , xmin =0, xmax = 5;
  else if (varname == "PtBalanceZ"  ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 2;
  else if (varname == "PtBalanceW"  ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 2;
  else if (varname == "ZetaLep"     ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 10;

  else if (varname == "Jet1E"       ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 1000;
  else if (varname == "Jet2E"       ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 500;
  else if (varname == "Jet3E"       ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 300;
  else if (varname == "Jet1Pt"      ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 300;
  else if (varname == "Jet2Pt"      ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 300;
  else if (varname == "Jet3Pt"      ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 200;
  else if (varname == "Jet1Eta"     ) title=varname, proj_str= varname, nbins = 50, xmin =-5, xmax = 5;
  else if (varname == "Jet2Eta"     ) title=varname, proj_str= varname, nbins = 50, xmin =-5, xmax = 5;
  else if (varname == "Jet3Eta"     ) title=varname, proj_str= varname, nbins = 50, xmin =-5, xmax = 5;

  else if (varname == "Lep1Pt"      ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 500;
  else if (varname == "Lep2Pt"      ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 500;
  else if (varname == "Lep3Pt"      ) title=varname, proj_str= varname, nbins = 50, xmin =0, xmax = 300;
  else if (varname == "Lep1Eta"      ) title=varname, proj_str= varname, nbins = 50, xmin =-2.5, xmax = 2.5;
  else if (varname == "Lep2Eta"      ) title=varname, proj_str= varname, nbins = 50, xmin =-2.5, xmax = 2.5;
  else if (varname == "Lep3Eta"      ) title=varname, proj_str= varname, nbins = 50, xmin =-2.5, xmax = 2.5;

  // Jet1Phi // Jet1Y           // Mt_WZ
  // Jet2Phi // Jet2Y           // M_Z
  // Jet3Phi // Jet3Y           

  vector<int> masses{0,200,300,400,500,600,700,800,900};

  TCanvas* c1 = new TCanvas ("name", "title", 800, 600);

  auto legend = new TLegend(0.12,0.12,0.25,0.4);
  legend->SetHeader("Mass (GeV)","C"); 

  for (auto mass : masses) {
    TH1F* hist = get_hist(mass);

    TString option="same hist";
    if (mass==0) option="hist";

    hist->Draw(option);
    char smass[3];
    if (mass != 0) { sprintf(smass, "%i", mass); }
    else { sprintf(smass, "%s", "bck"); }
    legend->AddEntry(hist,smass,"f");

  }

  if (varname=="pSignal") gPad->SetLogy();
  gStyle->SetOptStat(0);
  legend->Draw();
  c1->SaveAs("ControlPlots/"+idir+"/NN_output/"+varname+"_"+tmass+".png");
  c1->SaveAs("ControlPlots/"+idir+"/NN_output/"+varname+"_"+tmass+".root");

  return;
 
}

