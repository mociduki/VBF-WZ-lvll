#!/bin/bash 
:: Random hyper parameter optimization in batch mode for NN

source /home/zp/freund/.bashrc
. /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
. $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt ROOT" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt root_numpy" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt tensorflow" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt keras" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt pandas" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt matplotlib" \
       "lcgenv -p LCG_92python3 x86_64-slc6-gcc62-opt h5py"

numlayer=$(( ( RANDOM % 5 )  + 1 ))
numn=$(( ( ( RANDOM %30 )  + 1 ) * 10 ))
epochs=300
booldrop=$(( ( RANDOM % 2 ) ))

dropout=$(python3 -c "import random;print(random.randint(0, 60)*0.01)")
dropoutr=$(printf "%.2f" $dropout)
patience=$(( ( RANDOM % 20 )  + 1 )) 
lrrate=$(python3 -c "import random;print(random.randint(5, 80)*0.001)")
lrrater=$(printf "%.3f" $lrrate)
model=GM

name=${lrrater}_${numlayer}_${numn}_${booldrop}_${dropoutr}_${patience}_

python3 OPT_VBS_NN.py --v 2 \
    --lr $lrrater \
    --epoch $epochs \
    --numn $numn \
    --numlayer $numlayer \
    --booldrop $booldrop \
    --dropout $dropoutr \
    --patience $patience \
    --model $model \
    --output $name

exit 0