#!/bin/bash
export KERAS_BACKEND=tensorflow
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
. /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh

if [[ $HOSTNAME == *"lps.umontreal.ca"* ]];
then
    . $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt ROOT" \
	"lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt root_numpy" \
	"lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt tensorflow" \
	"lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt keras" \
	"lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt pandas" \
	"lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt matplotlib" \
	"lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt h5py" \
	"lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt scikitlearn"
else
    . $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "lcgenv -p LCG_97python3 x86_64-centos7-gcc8-opt ROOT" \
        "lcgenv -p LCG_97python3 x86_64-centos7-gcc8-opt root_numpy" \
        "lcgenv -p LCG_97python3 x86_64-centos7-gcc8-opt tensorflow" \
        "lcgenv -p LCG_97python3 x86_64-centos7-gcc8-opt keras" \
        "lcgenv -p LCG_97python3 x86_64-centos7-gcc8-opt pandas" \
        "lcgenv -p LCG_97python3 x86_64-centos7-gcc8-opt matplotlib" \
        "lcgenv -p LCG_97python3 x86_64-centos7-gcc8-opt h5py" \
        "lcgenv -p LCG_97python3 x86_64-centos7-gcc8-opt scikitlearn"
    #export PYTHONPATH=/cvmfs/sft.cern.ch/lcg/releases/numpy/1.18.2-7a597/x86_64-mac1015-clang110-opt/lib/python3.7/site-packages/numpy/core/init.py
fi
