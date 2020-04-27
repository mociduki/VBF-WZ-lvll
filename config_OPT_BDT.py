import numpy as np
class input_samples:
    #Assumed luminosity
    lumi = 140.
    #Fraction used for training
    valfrac = 0.7
    #Directory where ntuples are located
    filedir = './Input/'
    #Bkg Samples
    bckgr = {
        'name' : ['364253_Sherpa_222_NNPDF30NNLO_lllv_Systematics.root', 
                  '364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_Systematics.root'],
        'xs' : [4579., 47.],
        'nevents' : [75259300, 7325000]
    }

    #Signal Samples
    filedirsig = filedir + 'new/signalsNew/mc16d/'
    sigGM = {
        'name' : ['305029_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_300_qcd0_Systematics.root',
                  '305030_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_400_qcd0_Systematics.root',
                  '305031_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_500_qcd0_Systematics.root',
                  '305032_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_600_qcd0_Systematics.root',
                  '305033_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_700_qcd0_Systematics.root',
                  '305034_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_800_qcd0_Systematics.root',
                  '305035_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_900_qcd0_Systematics.root'],

        'xs' : [1.765,0.98095,0.6107,0.4028,0.2751,0.1934,0.1386,0.1010],
        'nevents' : [160000,160000,160000,160000,160000,160000,160000,160000]
    }
    #HVT signal files
    sigHVT = {
        'name' : ['MVA.307730_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0250_ntuples.root',
                  'MVA.307731_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0300_ntuples.root',
                  'MVA.307732_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0400_ntuples.root',
                  'MVA.307733_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0500_ntuples.root',
                  'MVA.307734_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0600_ntuples.root',
                  'MVA.307735_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0700_ntuples.root',
                  'MVA.307736_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0800_ntuples.root',
                  'MVA.307737_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0900_ntuples.root',
                  'MVA.307738_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m1000_ntuples.root',
                  'MVA.309528_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0350_ntuples.root',
                  'MVA.309529_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0450_ntuples.root'
                  ],
        'xs' : [24.42,10.54,2.299,0.7975,0.3408,0.1663,0.087984,0.049882,0.02961,1.3050,4.4843],
        'nevents' : [190000,190000,190000,190000,190000,190000,190000,175000,190000,160000,160000]
    }
    #Variables used for training
    variables = ['M_jj','Deta_jj','PtBalanceZ','PtBalanceW' ,'Jet1Pt', 'Jet2Pt', \
                 'Jet1Eta','Jet2Eta','Jet1E','Jet2E',\
                     'Lep1Eta','Lep2Eta', 'Lep3Eta','Lep1Pt', 'Lep2Pt','Lep3Pt','Pt_W',\
                     'Pt_Z','ZetaLep','Njets','Met']
#No 3rd jet
#                     'm_Pt_jet3',,'m_Eta_jet3''m_E_jet3',

#Contains list of samples to apply NN
class apply_samples:
    filedirbkg = './Input/'
    list_apply_bkg = ['364253_Sherpa_222_NNPDF30NNLO_lllv_Systematics.root',
                      '364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_Systematics.root']
    filedirsig = filedirbkg + 'new/signalsNew/mc16a/'
    list_apply_sigGM = ['MVA.305028_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_200_qcd0_ntuples.root',
                      'MVA.305029_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_300_qcd0_ntuples.root',
                      'MVA.305030_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_400_qcd0_ntuples.root',
                      'MVA.305031_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_500_qcd0_ntuples.root',
                      'MVA.305032_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_600_qcd0_ntuples.root',
                      'MVA.305033_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_700_qcd0_ntuples.root',
                      'MVA.305034_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_800_qcd0_ntuples.root',
                      'MVA.305035_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_900_qcd0_ntuples.root']
    list_apply_sigHVT = ['MVA.307730_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0250_ntuples.root',
                  'MVA.307731_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0300_ntuples.root',
                  'MVA.307732_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0400_ntuples.root',
                  'MVA.307733_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0500_ntuples.root',
                  'MVA.307734_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0600_ntuples.root',
                  'MVA.307735_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0700_ntuples.root',
                  'MVA.307736_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0800_ntuples.root',
                  'MVA.307737_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0900_ntuples.root',
                  'MVA.307738_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m1000_ntuples.root',
                  'MVA.309528_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0350_ntuples.root',
                  'MVA.309529_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0450_ntuples.root'
                  ]

    labelGM = np.arange(len(list_apply_sigGM))
    labelHVT = np.arange(len(list_apply_sigHVT))

