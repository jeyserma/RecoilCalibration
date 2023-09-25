
import numpy as np
import pickle
import logging

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gErrorIgnoreLevel = 6001

import utils
import quantileFitter as qf
import responseCorr as rc

def bkg_perp():
    proc = "bkg"
    baseDir = "%s/%s_perp" % (outDir, proc)
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['bkg_perp']

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100]
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 155, 5))
    quantiles = [0.0, 1e-5, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    
    # working
    recoilRebin = [-100, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 60, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150] # 0.015 equals around +/- 20 GeV
    
    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -45, -40, -35, -30, ] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150] # 0.015 equals around +/- 20 GeV

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, baseDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=recoilRebin)
    rqf.setQuantiles(-1)
    rqf.setSplineConfig([2, 6, 25, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.fit(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()


def bkg_para():
    proc = "bkg"
    baseDir = "%s/%s_para" % (outDir, proc)
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['bkg_para']

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100]
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 155, 5))
    quantiles = [0.0, 1e-5, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    
    recoilRebin = [-100, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 70, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150] # 0.015 equals around +/- 20 GeV

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, baseDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=recoilRebin)
    rqf.setQuantiles(-1)
    rqf.setSplineConfig([2, 6, 25, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.fit(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()



def zmumu_perp():
    proc = "zmumu"
    baseDir = "%s/%s_perp" % (outDir, proc)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['zmumu_perp']

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100]
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 155, 5))
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150] # 0.015 equals around +/- 20 GeV
    quantiles = [0.0, 1e-5, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    
    recoilRebin = [-100, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 60, 100]
    
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150] # 0.015 equals around +/- 20 GeV
    
    # orig
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100]
    #quantiles = [0.0, 0.75e-5, 1e-5, 0.25e-4, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, baseDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(-1, test=recoilRebin)
    rqf.setSplineConfig([2, 6, 25, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    #rqf.plot_refit_quantiles(prefit=False, logY=False)
    #rqf.qparms_postfit()


def zmumu_para():
    proc = "zmumu"
    baseDir = "%s/%s_para" % (outDir, proc)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['zmumu_para']


    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150] # 0.015 equals around +/- 20 GeV
    quantiles = [0.0, 1e-5, 0.8e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    recoilRebin = [-100, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 70, 100]
    # orig
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100]
    #quantiles = [0.0, 1e-5, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    
    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, baseDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=recoilRebin)
    rqf.setQuantiles(-1)
    rqf.setSplineConfig([2, 6, 25, 60], extrpl=[None, 60])
    
    #rqf.plot_refit_quantiles(ext=f"data/zmumu_para_{met}_postfit.pkl")
    

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()



def singlemuon_perp():
    proc = "singlemuon"
    baseDir = "%s/%s_perp" % (outDir, proc)
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['singlemuon_perp']
    bhist_bkg = data['bkg_perp']
    baseDir_bkg = baseDir.replace("singlemuon", "bkg")

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100]
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 155, 5))
    quantiles = [0.0,  1e-5, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    # working
    recoilRebin = [-100, -85, -70, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 70, 85, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150] # 0.015 equals around +/- 20 GeV
    
    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -45, -40, -35, -30, ] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150] # 0.015 equals around +/- 20 GeV
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, baseDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=recoilRebin)
    rqf.setQuantiles(-1)
    rqf.setSplineConfig([2, 6, 25, 60], extrpl=[None, 60])
    rqf.setBackground(f"data/bkg_perp_{met}_postfit.pkl", bhist_bkg, sf=1.0)

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()


def singlemuon_para():
    proc = "singlemuon"
    baseDir = "%s/%s_para" % (outDir, proc)
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['singlemuon_para']
    bhist_bkg = data['bkg_para']
    baseDir_bkg = baseDir.replace("singlemuon", "bkg")

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100]
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 155, 5))
    quantiles = [0.0, 1e-5, 1e-4, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 150, 5)) + [150]
    recoilRebin = [-100, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 70, 100]
    recoilRebin = [-100, -80, -70, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 60, 70, 80, 90, 100]
    
    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, baseDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=recoilRebin)
    rqf.setQuantiles(-1)
    rqf.setSplineConfig([2, 6, 25, 60], extrpl=[None, 60])
    rqf.setBackground(f"data/bkg_para_{met}_postfit.pkl", bhist_bkg, sf=1.0)

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()


def response_para():
    tag = "singlemuon_para_response"
    baseDir = "%s/%s" % (outDir, tag)
    proc = "singlemuon"
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    bhist_data = data['singlemuon_para']
    bhist_bkg = data['bkg_para']
    bhist = bhist_data + bhist_bkg*(-1)
    
    qTrebin = list(range(0, 150, 1)) + [150]
    corr = rc.ResponseCorr(proc, met, procLabel, metLabel, baseDir, lumiLabel)
    corr.setHist(bhist, qTrebin)
    corr.correct(knots=[0, 2, 4, 6, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 140], extrpl=[None, 140])



def wmunu_perp_gen():
    proc = "wmunu_gen"
    baseDir = "%s/%s_perp" % (outDir, proc)
    procLabel = "W^{#pm} #rightarrow #mu^{#pm}#nu (q_{T} gen)"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['wmunu_perp_gen']

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    #recoilRebin = list(range(-100, 102, 2))
    qTRebin = list(range(0, 150, 2)) + [150]
    quantiles = [0.0, 0.5e-5, 1e-5, 0.5e-4, 0.75e-4, 1e-4,  0.25e-3, 0.5e-3,  0.75e-3, 1e-3,  0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, baseDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(quantiles)
    rqf.setSplineConfig([2, 6, 35, 80], extrpl=[None, 80])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()

    rqf.plot_refit_quantiles(ext=f"data/zmumu_gen_perp_{met}_postfit.pkl")



def wmunu_para_gen():
    proc = "wmunu_gen"
    baseDir = "%s/%s_para" % (outDir, proc)
    procLabel = "W^{#pm} #rightarrow #mu^{#pm}#nu (q_{T} gen)"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['wmunu_para_gen']

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    #recoilRebin = list(range(-100, 102, 2))
    qTRebin = list(range(0, 150, 2)) + [150]
    quantiles = [0.0, 1e-5, 1e-4,  0.25e-3, 0.5e-3,  0.75e-3, 1e-3,  0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, baseDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(quantiles)
    rqf.setSplineConfig([2, 6, 25, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False)
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()

    rqf.plot_refit_quantiles(ext=f"data/zmumu_gen_para_{met}_postfit.pkl")



def zmumu_perp_gen():
    proc = "zmumu_gen"
    baseDir = "%s/%s_perp" % (outDir, proc)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus} (q_{T} gen)"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['zmumu_perp_gen']

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100]
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 155, 5))
    quantiles = [0.0, 0.75e-5, 1e-5, 0.25e-4, 0.5e-4, 0.75e-4, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, baseDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(quantiles)
    rqf.setSplineConfig([2, 6, 25, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False) # ext=f"data/zmumu_perp_{met}_postfit.pkl"
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()

    rqf.plot_refit_quantiles(ext=f"data/zmumu_perp_{met}_postfit.pkl")


def zmumu_para_gen():
    proc = "zmumu_gen"
    baseDir = "%s/%s_para" % (outDir, proc)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus} (q_{T} gen)"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(baseDir, False)
    bhist = data['zmumu_para_gen']

    recoilRebin = [-100, -80, -70, -65, -60, -55, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + [100]
    #qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 2)) + list(range(100, 155, 5))
    quantiles = [0.0, 1e-5, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, baseDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(quantiles)
    rqf.setSplineConfig([2, 6, 25, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False, ext=f"data/zmumu_para_{met}_postfit.pkl")
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()

    rqf.plot_refit_quantiles(ext=f"data/zmumu_para_{met}_postfit.pkl")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    met = "DeepMETReso"
    flavor = "mumu"
    lumiLabel = "16.8 fb^{#minus1} (13 TeV)"

    with open("data/recoil_DeepMETReso_highPU.pkl", "rb") as f:
        data = pickle.load(f)

    outDir = "/home/submit/jaeyserm/public_html/recoil/highPU/DeepMETReso/"
    utils.mkdir(outDir, False)

    #zmumu_para()
    zmumu_perp()

    #zmumu_para_gen()
    #zmumu_perp_gen()

    #wmunu_para_gen()
    #wmunu_perp_gen()

    #bkg_para()
    #bkg_perp()

    #singlemuon_para()
    #singlemuon_perp()

    #response_para()