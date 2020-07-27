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
#include <TLegend.h>
#include <TDirectory.h>
#include <iostream>
#include <unordered_map>

// SUBDIRECTORIES TO EDIT
string idir  = "0630/";
string tmass = "m900";
string sdir  = idir+tmass;

string get_file_name(int mass, string phys_model="GM") {
  
  string insert_str="main";
  string              file_path="OutputRoot/"+sdir+"/new_"+phys_model+"_"+insert_str;
  if      (mass==200) file_path+="MVA.450765_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m200_ntuples.root";
  else if (mass==250) file_path+="MVA.450766_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m250_ntuples.root";
  else if (mass==300) file_path+="MVA.450767_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m300_ntuples.root";
  else if (mass==350) file_path+="MVA.450768_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m350_ntuples.root";
  else if (mass==400) file_path+="MVA.450769_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m400_ntuples.root";
  else if (mass==450) file_path+="MVA.450770_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m450_ntuples.root";
  else if (mass==500) file_path+="MVA.450771_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m500_ntuples.root";
  else if (mass==600) file_path+="MVA.450772_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m600_ntuples.root";
  else if (mass==700) file_path+="MVA.450773_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m700_ntuples.root";
  else if (mass==800) file_path+="MVA.450774_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m800_ntuples.root";
  else if (mass==900) file_path+="MVA.305035_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_900_qcd0_ntuples.root";

  if (phys_model=="HVT") {
    file_path="OutputRoot/"+sdir+"/new_"+phys_model+"_"+insert_str;
    if      (mass== 250) file_path+="MVA.307730_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0250_ntuples.root";
    else if (mass== 300) file_path+="MVA.307731_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0300_ntuples.root";
    else if (mass== 350) file_path+="MVA.309528_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0350_ntuples.root";
    else if (mass== 400) file_path+="MVA.307732_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0400_ntuples.root";
    else if (mass== 450) file_path+="MVA.309529_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0450_ntuples.root";
    else if (mass== 500) file_path+="MVA.307733_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0500_ntuples.root";
    else if (mass== 600) file_path+="MVA.307734_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0600_ntuples.root";
    else if (mass== 700) file_path+="MVA.307735_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0700_ntuples.root";
    else if (mass== 800) file_path+="MVA.307736_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0800_ntuples.root";
    else if (mass== 900) file_path+="MVA.307737_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0900_ntuples.root";
    else if (mass==1000) file_path+="MVA.307738_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m1000_ntuples.root";
  }

  bool warning= (phys_model=="GM" and mass==1000) or (phys_model=="HVT" and mass==200);
  if (warning) {cout<<"phys_model: "<<phys_model<<" while mass= "<<mass<<". Aborting the process."<<endl; exit(1);}

  return file_path;
  
}

int get_color(int mass) {

  int color = kBlack;
  if      (mass==250) color=kGray+2;
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

TString title, proj_str,select_weight;
TString proj_option="";
int nbins = 50; float xmin =0, xmax = 1;

TH1F* get_bkg_hist(TString phys_model="GM") {

  TChain* chain = new TChain("nominal");
  TString ins_str="main";
  //chain->Add("OutputRoot/new_GM_"+ins_str+"MVA.364253_Sherpa_222_NNPDF30NNLO_lllv_ntuples.root");
  chain->Add(((TString)"OutputRoot/")+sdir.data()+"/new_"+phys_model+"_"+ins_str+"MVA.361292_MGaMcAtNloPy8EG_NNPDF30LO_A14NNPDF23LO_WZ_lvll_FxFx_ntuples.root");
  chain->Add(((TString)"OutputRoot/")+sdir.data()+"/new_"+phys_model+"_"+ins_str+"MVA.364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_ntuples.root");

  TH1F* hist = new TH1F("bkg",title,nbins,xmin,xmax);
  chain->Project(hist->GetName(),proj_str,select_weight,proj_option);

  return hist;
}

float mfac=20;

TH1F* get_hist(int mass,TString phys_model="GM") {

  TH1F* hist;
  if (mass>0) {
    string fname=get_file_name(mass,phys_model.Data());
    
    TFile* f = TFile::Open(fname.data(),"read");
    TTree* t = (TTree*)f->Get("nominal");
    
    TString histName = "mass"+TString::Itoa(mass,10);
    //std::cout<<histName<<std::endl;

    hist = new TH1F(histName ,title,nbins,xmin,xmax);
    select_weight+="*(abs(Weight)<10)";
    t->Project(hist->GetName(),proj_str,select_weight,proj_option);
  }
  else hist = get_bkg_hist(phys_model.Data());

  hist->SetMaximum(hist->GetBinContent( hist->GetMaximumBin() )*mfac);
  hist->SetMinimum(1e-3);
  hist->SetLineWidth(2);
  hist->SetLineColor(get_color(mass));
  
  return hist;
}

float AMS(float s, float b, bool debug=false) {

  if (s<=0 or b<=0) return 0;

  float br = 0.00001;// #KM: systematic unc?
  float sigma=sqrt(b+br);
  float n=s+b+br;
  float radicand = 2 *( n * log (n*(b+br+sigma)/(b*b+n*sigma+br))-b*b/sigma*log(1+sigma*(n-b)/(b*(b+br+sigma))));

  float ams= 0;
  if (radicand < 0) std::cout<<"AMS: radicand is negative. Returning 0."<<std::endl;
  else       ams= sqrt(radicand);

  if (debug) std::cout<<"s, b="<<s<<"\t"<<b<<", ams="<<ams<<std::endl;

  return ams;

}

TH1F* get_significance_hist(TH1F* h_sig, TH1F* h_bkg, float sf) {
  
  TString hname="significance_";
  TH1F* significance = (TH1F*) h_sig->Clone(hname+h_sig->GetName());
  significance->Reset();
  significance->SetTitle("Significance for yield / 140fb-1");

  h_sig->Scale(sf);
  h_bkg->Scale(sf);

  float Nsig=0,Nbkg=0;
  for( int i=0; i< significance->GetNbinsX(); i++) {
    Nsig=h_sig->Integral(i,h_sig->GetNbinsX());
    Nbkg=h_bkg->Integral(i,h_bkg->GetNbinsX());
    significance->SetBinContent(i,AMS(Nsig,Nbkg));
  }

  return significance;

}

void nn_per_mass(string dir="", string name="",TString varname="pSignal",bool norm2yield=false, TString phys_model="GM") {

  if (norm2yield) mfac=20;

  idir = dir;
  tmass = name;
  sdir  = idir+tmass;

  if      (varname == "pSignal"     ) title="NN output : "+tmass, proj_str=varname, nbins = 25, xmin =0, xmax = 1;
  else if (varname == "M_WZ"        ) title=varname, proj_str=varname, nbins = 25, xmin =0, xmax = 1500;
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

  select_weight = "(M_jj>100)";
  if (norm2yield) select_weight += "*WeightNormalized";
  else proj_option="norm"; //normalize to 1

  vector<int> masses{0,250,300,400,500,700,800,900};//600,

  TCanvas* c1 = new TCanvas ("name", "title", 800, 600);

  auto legend = new TLegend(0.7,0.65,0.9,0.9);
  legend->SetHeader("Mass (GeV)","C"); 
  legend->SetFillStyle(0); 
  legend->SetLineWidth(0); 
  legend->SetNColumns(2); 

  std::unordered_map<int,TH1F*> hists;

  for (auto mass : masses) {
    TH1F* hist = get_hist(mass,phys_model.Data());
    hists[mass]=hist;

    TString option="same hist";
    if (mass==0) option="hist";

    hist->Draw(option);
    char smass[3];
    if (mass != 0) { sprintf(smass, "%i", mass); }
    else { sprintf(smass, "%s", "bkg"); }
    legend->AddEntry(hist,smass,"f");

  }

  if (varname=="pSignal" and norm2yield) gPad->SetLogy();

  gStyle->SetOptStat(0);
  legend->Draw();

  string imagePath = "ControlPlots/"+idir+"/"+varname.Data() + (tmass!="" ? "_"+tmass : "");

  c1->SaveAs((imagePath+".png" ).data());
  c1->SaveAs((imagePath+".root").data());

  if (not (norm2yield and varname=="pSignal")) return;

  auto c2 = new TCanvas("c2","title",800,600);

  float sf= 1;

  for (auto mass : masses) {
    if      (mass==0 ) continue;

    auto significance = get_significance_hist(hists[mass],hists[0],sf);
    if (mass<=250) {
      significance->SetMaximum(significance->GetBinContent( significance->GetMaximumBin() )*2);//significance->SetMaximum(10);
      //significance->SetMinimum(1e-2);
    }
    TString option="same hist";
    if (mass==1) option="hist";

    significance->Draw(option);
    
  }
  legend->Draw();

  imagePath = "ControlPlots/"+idir+"/significance" + (tmass!="" ? "_"+tmass : "");

  c2->SaveAs((imagePath+".png" ).data());
  c2->SaveAs((imagePath+".root").data());

  return;
 
}

