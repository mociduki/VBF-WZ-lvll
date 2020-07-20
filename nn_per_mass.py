import ROOT
import sys

ROOT.gROOT.ProcessLine(".L nn_per_mass.C+")

if len(sys.argv)>1: ROOT.nn_per_mass(sys.argv[1],sys.argv[2])
else:               ROOT.nn_per_mass()
