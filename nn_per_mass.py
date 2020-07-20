import ROOT
import sys

ROOT.gROOT.ProcessLine(".L nn_per_mass.C+")

if   len(sys.argv)>3: ROOT.nn_per_mass(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
elif len(sys.argv)>2: ROOT.nn_per_mass(sys.argv[1],sys.argv[2],sys.argv[3])
elif len(sys.argv)>1: ROOT.nn_per_mass(sys.argv[1],sys.argv[2])
else:               ROOT.nn_per_mass()
