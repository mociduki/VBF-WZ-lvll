#!/bin/bash
export KERAS_BACKEND=tensorflow
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
. /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
. $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt ROOT" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt root_numpy" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt tensorflow" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt keras" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt pandas" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt matplotlib" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt h5py" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt scikitlearn"
