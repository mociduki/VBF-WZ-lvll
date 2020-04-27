#!/bin/bash 
:: Random hyper parameter optimization in batch mode for BDT

source /home/zp/freund/.bashrc
. /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
. $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt ROOT" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt root_numpy" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt tensorflow" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt keras" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt pandas" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt matplotlib" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt scikitlearn"

nest=$(( ( ( RANDOM % 20 )  + 1 ) * 100  ))
depth=$(( ( RANDOM % 5 )  + 1 ))
lr=$(python3 -c "import random;print(random.randint(0, 60)*0.001)")
opt=0
model=GM

name=${opt}_${nest}_${depth}_${lr}_

echo "==> Running the BDT Optimisation"

python3 OPT_VBS_BDT.py --v 2 \
    --opt $opt \
    --mode $model \
    --depth $depth \
    --lr $lr \
    --output $name

exit 0