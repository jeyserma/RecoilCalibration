
import sys,array,math,os
import numpy as np
import scipy
import copy
import pickle
import logging

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gErrorIgnoreLevel = 6001 

import utils
import quantileFitter as qf
import quantileFitter_R as qfr


 
def bkg_perp():

    tag = "bkg_perp"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['bkg_perp']

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    #recoilRebin = list(range(-100, 102, 2))
    qTRebin = list(range(0, 150, 2)) + [150]

    quantiles = [0.0, 0.5e-5, 1e-5, 0.5e-4, 0.75e-4, 1e-4,  0.25e-3, 0.5e-3,  0.75e-3, 1e-3,  0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    quantiles = [0.0, 1e-4,  1e-3,  0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    rqf = qf.QuantileFitter(procLabel, metLabel, baseDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=recoilRebin)
    rqf.setQuantiles(quantiles)
    rqf.setSplineConfig([2, 6, 35, 80], extrpl=[None, 80])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.fit(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()


def bkg_para():
    tag = "bkg_para"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    
    bhist_ttbar = readProc(groups, "recoil_corr_xy_para_qTbinned", ["TTbar"])
    bhist_ewk = readProc(groups, "recoil_corr_xy_para_qTbinned", ["EWK"])
    bhist = bhist_ewk + bhist_ttbar
    
    # ORIG
    recoilRebin = [-100, -90, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    quantiles = [0.0, 1e-6, 1e-5, 0.25e-4, 0.5e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    # NEW
    recoilRebin = [-100, -90, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    quantiles = [0.0, 1e-6, 1e-5, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    
    #recoilRebin = [-100, -90, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    #quantiles = [0.0, 1e-5, 0.5e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    recoilRebin = [-100, -90, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    quantiles = [0.0, 0.5e-5, 0.75e-5, 1e-5, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    

    
    rqf = qf.QuantileFitter(procLabel, metLabel, baseDir, "para", lumiLabel)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebin=recoilRebin)
    rqf.setQuantiles(quantiles)
    rqf.setFitOrder(4)

    rqf.parameterizeQuantiles()
    rqf.paramsVsQuantile()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.refitQuantiles(withConstraint=True)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
   
def bkg_para_old():
    tag = "bkg_para"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    
    bhist_ttbar = readProc(groups, "recoil_corr_xy_para_qTbinned", ["TTbar"])
    bhist_ewk = readProc(groups, "recoil_corr_xy_para_qTbinned", ["EWK"])
    bhist = bhist_ewk + bhist_ttbar
    
    recoilRebin = [-100, -90, -80, -75, -70, -65, -60, -55] + list(range(-50, 50, 2)) + [50, 55, 60, 65, 70, 75, 80, 90, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150] # + list(range(100, 150, 25)) + [150]
    quantiles = [0.0,  1e-5, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    quantiles = [0.0, 1e-6, 0.75e-5, 1e-5, 0.25e-4, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    # 
    rqf = qf.QuantileFitter(procLabel, metLabel, baseDir, "para", lumiLabel)
    rqf.setHistConfig(bhist, "recoil_para", 0, 150, min(recoilRebin), max(recoilRebin), qTRebin=4, recoilRebin=recoilRebin)
    rqf.setQuantiles(quantiles)
    rqf.setFitOrder(3)

    rqf.parameterizeQuantiles()
    rqf.paramsVsQuantile()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.refitQuantiles()
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)


def zmumu_perp():
    tag = "zmumu_perp"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['zmumu_perp']

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    #recoilRebin = list(range(-100, 102, 2))
    qTRebin = list(range(0, 150, 2)) + [150]

    quantiles = [0.0, 0.5e-5, 1e-5, 0.5e-4, 0.75e-4, 1e-4,  0.25e-3, 0.5e-3,  0.75e-3, 1e-3,  0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    rqf = qf.QuantileFitter(procLabel, metLabel, baseDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=recoilRebin)
    rqf.setQuantiles(quantiles)
    rqf.setSplineConfig([2, 6, 35, 80], extrpl=[None, 80])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.fit(withConstraint=True)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()




def zmumu_perp_old():
    procs = ["Zmumu"]
    tag = "zmumu_perp"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(baseDir, False)
    bhist = readProc(groups, "recoil_corr_xy_perp_qTbinned", procs)
    
    recoilRebin = [-100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    quantiles = [0.0, 1e-6, 0.5e-5, 1e-5, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
       
    rqf = qf.QuantileFitter(procLabel, metLabel, baseDir, "perp", lumiLabel)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebin=recoilRebin)
    rqf.setQuantiles(quantiles)
    rqf.setFitOrder(3)

    rqf.parameterizeQuantiles()
    rqf.paramsVsQuantile()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.refitQuantiles(withConstraint=False)
    rqf.refitQuantiles(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)





def zmumu_para_qfr():
    tag = "zmumu_para"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    
    bhist = data['zmumu_para']
    
    quant_recvals = [-80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80]
    #quant_recvals = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    #quant_recvals = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    quant_recvals = [-80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80]
    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34, -32, -30, -28, -26, -24, -22] + list(range(-20, 20, 1)) + [20, 22, 24, 26, 28, 30, 32, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 100]
    
    qTRebin = list(range(0, 50, 2)) +  list(range(50, 100, 10)) + [100, 110, 120, 130, 140, 150]
    qTRebin = list(range(0, 50, 1)) +  list(range(50, 100, 10)) + [100, 110, 120, 130, 140, 150]
    qTRebin = list(range(0, 100, 2)) +  [100]

    rqf = qfr.QuantileFitter(procLabel, metLabel, baseDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(quant_recvals), max(quant_recvals), qTRebin=qTRebin, recoilRebin=1)
    rqf.setQuantiles(quant_recvals)
    rqf.setqTInterpolator("spline", [0, 5, 10,  50, 100])
    #rqf.setqTInterpolator("spline", [0,  10, 50, 150])
    #rqf.setqTInterpolator("poly", 5)

    rqf.parameterizeQuantiles()
    #rqf.paramsVsQuantile()
    #rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.refitQuantiles(withConstraint=False)
    #rqf.refitQuantiles(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, recoilRebin=quant_recvals)
    rqf.plot_coeff_recoil()
    #rqf.plot_refit_quantiles(prefit=False, logY=False)
        
        
def zmumu_para():
    tag = "zmumu_para"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    
    bhist = data['zmumu_para']
    
    recoilRebin = [-100, -80, -70, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 70,  80, 100]
    #recoilRebin =  list(range(-100, 102, 2))
    qTRebin = list(range(0, 150, 1)) + [150]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 1)) + list(range(100, 150, 2)) + [150]
    quantiles = [0.0, 0.25e-5, 0.5e-5, 0.75e-5, 1e-5, 0.25e-4, 0.5e-4, 0.75e-4,  1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075,  0.01, 0.02, 0.03, 0.04,  0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    quantiles_hi = [0.0, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075,  0.01, 0.015, 0.02, 0.025, 0.03, 0.04,  0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    # works with linear
    quantiles = [0.0, 0.1e-6, 0.35e-5, 0.75e-5,  1e-5, 0.25e-4, 0.5e-4, 0.75e-4,  1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005,  0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20,  0.30, 0.40, 0.50]
    quantiles_hi = [0.0, 0.5e-3, 0.8e-3, 1.1e-3, 0.0025, 0.00375,  0.005, 0.0075,  0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225,  0.30, 0.40, 0.50]
    #quantiles_hi = quantiles


    # stable config
    quantiles = [0.0, 1e-6,  1e-5,   1e-4,  0.5e-3, 0.75e-3, 0.875e-3, 1e-3,  0.0025, 0.005,  0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    quantiles_hi = [0.0, 1e-4, 0.5e-3, 0.75e-3, 1e-3,  0.0025, 0.00375, 0.005, 0.0075,  0.01, 0.015, 0.02,  0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    
    quantiles = [0.0, 1e-6,  1e-5,  1e-4, 0.25e-3,   0.5e-3,  0.75e-3, 0.875e-3, 1e-3,  0.0025, 0.005,  0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    quantiles_hi = [0.0, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3,  0.0025, 0.00375, 0.005, 0.0075,  0.01, 0.015, 0.02,  0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    
    quantiles = [0.0, 1e-6,  1e-5,  1e-4, 0.25e-3,   0.5e-3,  0.75e-3,  1e-3,  0.0025, 0.005,  0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    quantiles_hi = [0.0, 1e-4, 0.125e-3, 0.25e-3,   0.5e-3, 0.75e-3, 0.875e-3, 1e-3,  0.0025, 0.005,  0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    
    
    
    # perp config
    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    #recoilRebin = list(range(-100, 102, 2))
    qTRebin = list(range(2, 50, 2)) + list(range(50, 100, 2)) + list(range(100, 150, 2)) + [150]
    
    ## ok best
    quantiles = [0.0, 1e-6, 0.5e-5, 1e-5,0.25e-4,  0.5e-4,  1e-4, 0.25e-3, 0.5e-3, 1e-3,  0.0025, 0.005, 0.00625,  0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    quantiles_hi = [0.0, 0.5e-4, 0.25e-3, 0.5e-3, 0.625e-3, 0.75e-3, 0.875e-3, 1e-3, 0.00125, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.045, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    # try  again with uniform quantiles
    qTRebin = list(range(0, 150, 2)) + [150]
    quantiles = [0.0, 1e-5, 1e-4,  0.25e-3, 0.5e-3,  0.75e-3, 1e-3,  0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    quantiles_hi = quantiles 

    rqf = qf.QuantileFitter(procLabel, metLabel, baseDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=recoilRebin)
    rqf.setQuantiles(quantiles, quantiles_hi=quantiles_hi)
    rqf.setqTInterpolator("cspline", [2, 6, 15, 40, 80])
    
    rqf.qparms_prefit()
    #rqf.paramsVsQuantile()
    rqf.plot_refit_quantiles(prefit=True)
    #quit()
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms) # 
    rqf.plot_refit_quantiles(prefit=False)
    #rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()


        
        

def zmumu_para_old():
    tag = "zmumu_para"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    
    bhist = data['zmumu_para']
    
    
    recoilRebin = [-100, -90, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    quantiles = [0.0, 1e-6, 0.75e-5, 1e-5, 0.25e-4, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    #recoilRebin = [-100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    #quantiles = [0.0, 1e-6, 0.5e-5, 1e-5, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
 
    #recoilRebin = [-100, -90, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    #quantiles = [0.0, 0.5e-5, 1e-5, 0.5e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
 
    recoilRebin = [-100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    quantiles = [0.0, 1e-7, 0.5e-6, 1e-6, 0.5e-5, 1e-5, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    # OK quantiles = [0.0, 1e-5, 1e-4, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    # 0.5e-5 0.75e-5 nok
    # OK quantiles = [0.0, 0.75e-5, 0.5e-4, 1e-4, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    #quantiles = [0.0, 1e-6, 0.5e-5, 1e-5, 0.5e-4, 1e-4, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
 
    recoilRebin = [-100, -90, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    quantiles = [0.0, 1e-6, 0.5e-5, 0.75e-5, 1e-5, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    recoilRebin = [-100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    
    recoilRebin = [-100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34, -32, -30, -28, -26, -24, -22] + list(range(-20, 20, 1)) + [20, 22, 24, 26, 28, 30, 32, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    

    
    nquants = 42
    #lin = np.linspace(-3.2, 0, nquants) # 3.2
    #gauss_space =  0.5*(scipy.special.erf(lin)+1)
    #quantiles = [0] + list(gauss_space)
    #print(quantiles)
    
    #interp_sigmas = np.linspace(-4.0, 4.0, nquants)
    interp_sigmas = np.linspace(-4.6, 4.3, nquants)
    interp_cdfvals = 0.5*(1. + scipy.special.erf(interp_sigmas/np.sqrt(2.)))
    interp_cdfvals = np.concatenate([[0.], interp_cdfvals, [1.]])
    

    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 120, 5)) + [120, 130, 140, 150]
    recoilRebin = [-100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -36, -34, -32, -30, -28, -26] + list(range(-25, 25, 1)) + [25, 26, 28, 30, 32, 34, 36, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    import hist
    s = hist.tag.Slicer()
    left = bhist[{"qTbinned": s[complex(0,0):complex(0,150)], "recoil_para": s[complex(0,-500):complex(0,-100)]}]
    right = bhist[{"qTbinned": s[complex(0,0):complex(0,150)], "recoil_para": s[complex(0,100):complex(0,500)]}]
    
    
    ## working
    ##quantiles_perp = [0.0, 0.5e-5, 0.75e-5, 1e-5, 0.25e-4, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    ##qTRebin = list(range(0, 50, 2)) + [50] # + list(range(50, 100, 2)) # + list(range(100, 120, 5)) # + [120, 130, 140, 150]
    ##recoilRebin = [-80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80]
    
    sigma_left = -1.*scipy.special.erfinv(1.-right.sum().value/bhist.sum().value)*np.sqrt(2.)
    sigma_right = scipy.special.erfinv(1.-left.sum().value/bhist.sum().value)*np.sqrt(2.)

    nquants = 41
    interp_sigmas = np.linspace(-4.5, 4.5, nquants)
    #interp_sigmas = np.linspace(sigma_left, sigma_right, nquants)
    interp_cdfvals = 0.5*(1. + scipy.special.erf(interp_sigmas/np.sqrt(2.)))
    interp_cdfvals = np.concatenate([[0.], interp_cdfvals, [1.]])

    quantiles_perp = [0.0, 0.5e-5, 1e-5, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    quantiles_perp = [0.0, 0.25e-4, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    # works entire range quantiles_perp = [0.0, 0.25e-3,  0.5e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    # working
    quantiles_perp = [0.0, 0.75e-5, 1e-5, 0.25e-4, 0.5e-4, 0.75e-4,  1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005,  0.01, 0.02, 0.03,  0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    #quantiles_perp = [1e-4, 0.25e-3,0.5e-3, 1e-3, 0.0025, 0.005,  0.01, 0.02, 0.03,  0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    qTRebin = list(range(0, 50, 2)) +  list(range(50, 100, 5)) + [100, 110, 120, 130, 140, 150]
    
    #quantiles_perp = quantiles_perp + [1.-q for q in reversed(quantiles_perp[:-1])] # symmetrize
    #del quantiles_perp[1]
    
    #quantiles_perp = [0.0, 0.25e-3,  0.5e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    #quantiles_perp = [0.0, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    qTRebin = list(range(0, 50, 2)) + list(range(50, 70, 4)) + list(range(70, 100, 10)) + [100, 110, 120, 130, 150] # + [120, 130, 140, 150]  # + list(range(50, 100, 2)) # + list(range(100, 120, 5)) # + [120, 130, 140, 150]
    qTRebin = list(range(0, 140, 2)) + [140]

    qTRebin = list(range(0, 50, 1)) + list(range(50, 80, 2)) + list(range(80, 100, 5)) + [100, 120, 150] # +  list(range(50, 100, 5)) + [100, 110, 120, 130, 140, 150]
    #recoilRebin = [-100, -90, -85, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -36, -34, -32, -30, -28, -26] + list(range(-25, 30, 1)) + [30, 32, 34, 36, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100]
    recoilRebin = [ -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    
    #recoilRebin = [-90, -80, -75, -70, -68, -66, -64, -62, -60, -58, -56, -54, -52] + list(range(-50, 50, 2)) + [50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 75, 80, 100]
    rqf = qf.QuantileFitter(procLabel, metLabel, baseDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=4, recoilRebin=1)
    rqf.setQuantiles(quantiles_perp)
    rqf.setqTInterpolator("spline", [0, 5, 10, 25, 50, 100, 140])
    #rqf.setqTInterpolator("spline", [0, 5, 10, 15])
    #rqf.setqTInterpolator("poly", 6)

    rqf.parameterizeQuantiles()
    #rqf.paramsVsQuantile()
    #rqf.plot_refit_quantiles(prefit=True)
    rqf.refitQuantiles(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False, recoilRebin=recoilRebin)
    #rqf.plot_refit_quantiles(prefit=False, logY=False)
        
        
    


    
def singlemuon_perp():
    tag = "singlemuon_perp"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['singlemuon_perp']
    bhist_bkg = data['bkg_perp']
    baseDir_bkg = baseDir.replace("singlemuon", "bkg")

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    #recoilRebin = list(range(-100, 102, 2))
    qTRebin = list(range(0, 150, 2)) + [150]

    # now fix more tails...
    #quantiles = [0.0, 1e-6, 0.5e-5, 1e-5,0.25e-4,  0.5e-4,  1e-4, 0.25e-3, 0.5e-3, 1e-3,  0.0025, 0.005, 0.00625,  0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    quantiles = [0.0, 0.5e-5, 1e-5, 0.5e-4, 0.75e-4, 1e-4,  0.25e-3, 0.5e-3,  0.75e-3, 1e-3,  0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    #### interesting
    quantiles = [0.0, 1e-4,  1e-3,  0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50] ### interesting config for DATA only, no bkg. Extreme tails good, inner tails bad

    quantiles = [0.0, 1e-4,  1e-3,  0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    quantiles = [0.0, 0.25e-4, 1e-4,  0.25e-3, 0.5e-3,  0.75e-3, 1e-3,  0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    rqf = qf.QuantileFitter(procLabel, metLabel, baseDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=recoilRebin)
    rqf.setQuantiles(quantiles)
    rqf.setSplineConfig([2, 6, 35, 80], extrpl=[None, 80])
    rqf.setBackground(f"{baseDir_bkg}/postfit.pkl", bhist_bkg, sf=1.0)

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.fit(withConstraint=True, usePrevPostFit=False)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()




def singlemuon_para():
    procs = ["Data"]
    tag = "singlemuon_para"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    
    bhist = readProc(groups, "recoil_corr_xy_para_qTbinned", procs)
    bhist_ttbar = readProc(groups, "recoil_corr_xy_para_qTbinned", ["TTbar"])
    bhist_ewk = readProc(groups, "recoil_corr_xy_para_qTbinned", ["EWK"])
    bhist_bkg = bhist_ttbar + bhist_ewk
    baseDir_bkg = baseDir.replace("singlemuon", "bkg")
    
    recoilRebin = [-100, -90, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    quantiles = [0.0, 1e-6, 0.75e-5, 1e-5, 0.25e-4, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    quantiles = [0.0, 1e-6, 1e-5, 0.25e-4, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
      
    # NEW
    recoilRebin = [-100, -90, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    quantiles = [0.0, 1e-6, 1e-5, 0.5e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
      
    # NEW
    #recoilRebin = [-100, -90, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    #quantiles = [0.0,  1e-5, 0.5e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
  
    recoilRebin = [-100, -90, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    quantiles = [0.0, 0.5e-5, 0.75e-5, 1e-5, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
      
    rqf = qf.QuantileFitter(procLabel, metLabel, baseDir, "para", lumiLabel)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebin=recoilRebin) 
    rqf.setQuantiles(quantiles)
    rqf.setFitOrder(4)
    rqf.setBackground(f"{baseDir_bkg}/quantiles_refit/results_param.json", bhist_bkg, sf=1.0)
    
    rqf.parameterizeQuantiles()
    rqf.paramsVsQuantile()
    rqf.plot_refit_quantiles(prefit=True)
    #parms = rqf.refitQuantiles(withConstraint=False)
    #rqf.refitQuantiles(withConstraint=True, parms=parms)
    rqf.refitQuantiles(withConstraint=True)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
  

def singlemuon_para_old():
    procs = ["Data"]
    tag = "singlemuon_para"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    
    bhist = readProc(groups, "recoil_corr_xy_para_qTbinned", procs)
    bhist_ttbar = readProc(groups, "recoil_corr_xy_para_qTbinned", ["TTbar"])
    bhist_ewk = readProc(groups, "recoil_corr_xy_para_qTbinned", ["EWK"])
    bhist_bkg = bhist_ttbar + bhist_ewk
    baseDir_bkg = baseDir.replace("singlemuon", "bkg")
    
    recoilRebin = [-100, -90, -80, -75, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    quantiles = [0.0, 1e-6, 0.75e-5, 1e-5, 0.25e-4, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    
    rqf = qf.QuantileFitter(procLabel, metLabel, baseDir, "para", lumiLabel)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebin=recoilRebin) 
    rqf.setQuantiles(quantiles)
    rqf.setFitOrder(3)
    rqf.setBackground(f"{baseDir_bkg}/quantiles_refit/results_param.json", bhist_bkg, sf=1.0)
    
    rqf.parameterizeQuantiles()
    rqf.paramsVsQuantile()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.refitQuantiles()
    rqf.plot_refit_quantiles(prefit=False)
    #rqf.plot_refit_quantiles(prefit=False, logY=False)
 

  



    

    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    met = "DeepMETReso"
    flavor = "mumu"
    lumiLabel = "16.8 fb^{#minus1} (13 TeV)"

    with open("data/recoil_mz_wlike_with_mu_eta_pt_DeepMETReso.pkl", "rb") as f:
        data = pickle.load(f)

    outDir = "/home/submit/jaeyserm/public_html/recoil/highPU/mumu_DeepMETReso/"
    utils.mkdir(outDir, False)
    
    

    quantiles_perp = [0.0, 1e-6, 1e-5, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    quantiles_perp = [0.0, 0.5e-5, 1e-5, 0.5e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    #zmumu_para()
    #zmumu_perp()
        
    #bkg_para()
    #bkg_perp()


    #singlemuon_para()
    #singlemuon_perp()
    
    qf.exportModel(f"{outDir}/zmumu_perp/postfit.pkl", f"{outDir}/singlemuon_perp/postfit.pkl", "test.tflite")
