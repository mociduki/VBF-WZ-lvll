import numpy as np
import os
Filedir    = 'Inputs/MVA' #to change input dataset, change the link in the Inputs directory

class input_samples:
    #Assumed luminosity
    lumi = 140.
    #Fraction used for training 
    trafrac = 0.9
    #Directory where ntuples are located
    filedir = Filedir+"/"+"resonance."
    #Bkg Samples
    bckgr = {
        'name' : [ #'MVA.364253_Sherpa_222_NNPDF30NNLO_lllv_ntuples.root', # 4579.,75259300
                   #'364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_ntuples.root',
                   '361292_MGaMcAtNloPy8EG_NNPDF30LO_A14NNPDF23LO_WZ_lvll_FxFx_ntuples.root',
                   '364739_MGPy8EG_NNPDF30NLO_A14NNPDF23LO_lvlljjEW6_OFMinus_ntuples.root',
                   '364740_MGPy8EG_NNPDF30NLO_A14NNPDF23LO_lvlljjEW6_OFPlus_ntuples.root',
                   '364741_MGPy8EG_NNPDF30NLO_A14NNPDF23LO_lvlljjEW6_SFMinus_ntuples.root',
                   '364742_MGPy8EG_NNPDF30NLO_A14NNPDF23LO_lvlljjEW6_SFPlus_ntuples.root'
                   ],

        'xs' : [1704., 47.],
        'nevents' : [3890000, 7325000]
    }

    #Signal Samples
    filedirsig = filedir #+ 'new/signalsNew/mc16d/'
    #GM signal files
    sigGM = {
        'name' : ['450765_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m200_ntuples.root',
                  '450766_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m250_ntuples.root',
                  '450767_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m300_ntuples.root', #down to here it's working for all mass points
                  '450768_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m350_ntuples.root', #sesms like the threshold is here now
                  '450769_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m400_ntuples.root',
                  '450770_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m450_ntuples.root',
                  '450771_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m500_ntuples.root',
                  '450772_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m600_ntuples.root', #'MVA.305032_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_600_qcd0_ntuples.root',
                  '450773_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m700_ntuples.root',
                  '450774_MGaMcAtNloPy8EG_A14NNPDF23LO_vbfGM_sH05_H5pWZ_lvll_m800_ntuples.root',
                  '305035_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_900_qcd0_ntuples.root'
                  ],
        # mass           200,    250,     300,    350,    400,    450,    500,    600,    700,    800,    900
        #'xs'      : [1.765 , 1.4911, 0.98095, 0.8874, 0.6107, 0.6391, 0.4028, 0.2751, 0.1934, 0.1386, 0.1010],
        #'nevents' : [160000, 160000,  160000, 280000, 160000, 160000, 160000, 160000,  160000,  160000,  160000 ],
        'xs'      : [7.0596,  7.710,  3.9238,  4.582, 2.4428,  2.95, 1.6113, 1.1005, 0.77398, 0.55433, 0.40394 ],
        'nevents' : [ 70000,  70000,   70000, 190000,  70000,  70000,  70000,  70000,   70000,   70000,   50000 ],
        'filtEff' : [     1,0.77156,       1,0.77507,      1,0.77891,      1,      1,       1,       1,       1 ],
        'switch'  : [     1,      1,       1,      1,      0,      0,      0,      0,       0,       0,       0 ]
    }
    #HVT signal files
    sigHVT = {
        'name' : ['307730_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0250_ntuples.root',
                  '307731_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0300_ntuples.root',
                  '309528_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0350_ntuples.root',
                  '307732_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0400_ntuples.root',
                  '309529_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0450_ntuples.root',
                  '307733_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0500_ntuples.root',
                  '307734_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0600_ntuples.root',
                  '307735_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0700_ntuples.root',
                  '307736_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0800_ntuples.root',
                  '307737_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m0900_ntuples.root',
                  '307738_MGPy8EG_A14NNPDF23LO_vbfHVT_Agv1_VzWZ_lvll_m1000_ntuples.root',
                  ],
        #               250,    300,    350,    400,    450,      500,      600,     700,    800,    900,   1000
        'xs'      : [24.42 , 10.54 , 2.299 , 0.7975, 0.3408, 0.1663, 0.087984, 0.049882, 0.02961, 1.3050, 4.4843],
        'nevents' : [190000, 190000, 190000, 190000, 190000, 190000,   190000,   175000,  190000, 160000, 160000],
        'filtEff' : [     1,      1,      1,      1,      1,      1,       1,         1,       1,      1,      1],
        'switch'  : [     1,      1,      1,      1,      0,      0,        0,        0,       0,      0,      0]
    }
    #Variables used for training
    variables = ['M_jj','Deta_jj', 'Dphi_jj',
                 'Jet1Pt', 'Jet2Pt',
                 'Jet1Eta','Jet2Eta',
                 'Jet1E','Jet2E',
                 'Jet3Eta', #added
                 #'Lep1Pt', 'Lep2Pt','Lep3Pt', ### removed to reduce mass dependence
                 'Lep1Eta','Lep2Eta', 'Lep3Eta', 
                 'PtBalanceZ','PtBalanceW',
                 #'Pt_W', 'Pt_Z',  ### removed to reduce mass dependence
                 'Eta_W', 'Eta_Z', ### added instead
                 'ZetaLep','Njets' #,'Met'
                 ]

    #original set
    #variables = ['M_jj','Deta_jj',
    #             'Jet1Pt', 'Jet2Pt', 'Jet1Eta','Jet2Eta', 'Jet1E','Jet2E',
    #             'Lep1Pt', 'Lep2Pt', 'Lep3Pt', 'Lep1Eta','Lep2Eta','Lep3Eta',
    #             'PtBalanceZ','PtBalanceW',
    #             'Pt_W', 'Pt_Z', 
    #             'ZetaLep','Njets','Met']

#Contains list of samples to apply NN
class apply_samples:
    filedirapp=Filedir+"/"
    
    # Signal files
    list_apply_sigGM = input_samples.sigGM['name']

    list_apply_sigHVT = input_samples.sigHVT['name']
    
    # parse all files in the directory, except signals
    list_apply_bkg = []

    shortList= [450765,450766,450767,450768,450769,450770,450771,450772,450773,450774]
#    shortList= [450765,450766,450767,450768,450769,450770,450771,450772,450773,450774, #GM  sig
#                305032,305035,#305028,305029,305030,305031,305032,305033,305034,305035, #GM old, only include 600 & 900
#                307730,307731,307732,307733,307734,307735,307736,307737,307738,               #HVT sig
#                361292,364284]                                                                #WZ bkg
    useShortList=True
    if not useShortList: shortList=list()
    
    for r,d,f in os.walk(filedirapp):
        #print(f)
        for file in f:
            if 'history' in file: continue

            skipFlag=True
            for ch in shortList:
                if "{}".format(ch) in file: 
                    skipFlag=False
                    break
                pass
            if skipFlag: continue

            if '.root' in file:
                #print(file)
                #if file in list_apply_sigGM: continue
                #elif file in list_apply_sigHVT: continue
                #else: list_apply_bkg.append(file)
                
#                if   file=="MVA.364253_Sherpa_222_NNPDF30NNLO_lllv_ntuples.root" or   file=="MVA.364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_ntuples.root": 
#                    print(file)
#                    exit(1)
#                    pass

                list_apply_bkg.append(file)
            pass
        pass
    #print(list_apply_bkg) #exit(0)

    # data files
    # list_apply_data = []
    # for r,d,f in os.walk(filedirdata):
    #     for file in f:        
    #         if '.root' in file:
    #             list_apply_data.append(file)
    #             pass
    #         pass
    #     pass

    labelGM = np.arange(len(list_apply_sigGM))
    labelHVT = np.arange(len(list_apply_sigHVT))
