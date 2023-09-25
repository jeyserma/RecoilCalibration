
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
import responseCorr as rc

def bkg_perp():
    proc = "bkg"
    plotDir = "%s/%s_perp" % (basePlotDir, proc)
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(plotDir, False)
    bhist = data['bkg_perp']

    recoilRebin = [-100, -85,  -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 85, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    recoilRebin = [ -110, -85,  -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 85, 110]
    bins_for_quantiles = [-100, -85,  -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 85, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.fit(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False, xMin=-100, xMax=100)
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=-100, xMax=100)
    rqf.qparms_postfit()


def bkg_para():
    proc = "bkg"
    plotDir = "%s/%s_para" % (basePlotDir, proc)
    procLabel = "TTbar+EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(plotDir, False)
    bhist = data['bkg_para']

    recoilRebin = [-100,  -85, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    recoilRebin = [ -100,  -85, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    bins_for_quantiles = [-100,  -85, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    rqf.fit(withConstraint=False)
    rqf.plot_refit_quantiles(prefit=False, xMin=-100, xMax=100)
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=-100, xMax=100)
    rqf.qparms_postfit()


def zmumu_perp():
    proc = "zmumu"
    plotDir = "%s/%s_perp" % (basePlotDir, proc)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(plotDir, False)
    bhist = data['zmumu_perp']

    recoilRebin = [-150, -120, -100, -85,  -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 85, 100, 120, 150]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]
    
    recoilRebin = [-130, -110, -100, -95, -90, -85, -80, -75,  -70, -65, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 130]
    recoilRebin = [-130, -110] + list(range(-100, 100, 2)) + [100, 110, 130]
    bins_for_quantiles = recoilRebin
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]
    #quantiles = [0.0, 1e-5, 1e-4, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 0.0025, 0.005, 0.0075, 0.01, 0.015,0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    #rqf.setQuantiles(quantiles=quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    #rqf.qparms_prefit()
    #rqf.plot_refit_quantiles(prefit=True)
    #parms = rqf.fit(withConstraint=False)
    #rqf.fit(withConstraint=True, parms=parms)
    #rqf.plot_refit_quantiles(prefit=False, xMin=-130, xMax=130)
    #rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=-130, xMax=130)
    #rqf.qparms_postfit()
    rqf.knots_vs_cdfvals()


def zmumu_para():
    proc = "zmumu"
    plotDir = "%s/%s_para" % (basePlotDir, proc)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(plotDir, False)
    bhist = data['zmumu_para']

    recoilRebin = [-150, -100,  -85, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 150]
    bins_for_quantiles = [-100,  -85, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    #rqf.qparms_prefit()
    #rqf.plot_refit_quantiles(prefit=True)
    #parms = rqf.fit(withConstraint=False)
    #rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, xMin=-100, xMax=100)
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=-100, xMax=100)
    rqf.qparms_postfit()


def singlemuon_perp():
    proc = "singlemuon"
    plotDir = "%s/%s_perp" % (basePlotDir, proc)
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(plotDir, False)
    bhist = data['singlemuon_perp']
    bhist_bkg = data['bkg_perp']
    plotDir_bkg = plotDir.replace("singlemuon", "bkg")

    recoilRebin = [-100, -85,  -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 85, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    recoilRebin = [-110, -85,  -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 85, 110]
    bins_for_quantiles = [-100, -85,  -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 85, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])
    rqf.setBackground(f"{dataDir}/bkg_perp_postfit.pkl", bhist_bkg, sf=1.0)

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/zmumu_perp_postfit.pkl")
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, xMin=-100, xMax=100)
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=-100, xMax=100)
    rqf.qparms_postfit()


def singlemuon_para():
    proc = "singlemuon"
    plotDir = "%s/%s_para" % (basePlotDir, proc)
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(plotDir, False)
    bhist = data['singlemuon_para']
    bhist_bkg = data['bkg_para']
    plotDir_bkg = plotDir.replace("singlemuon", "bkg")

    recoilRebin = [-100, -90, -80, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]
    
    recoilRebin = [-100, -80, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]

    recoilRebin = [ -100,  -85, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    bins_for_quantiles = [-100,  -85, -70, -60, -55, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    qTRebin = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + [100]


    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=2)
    rqf.setQuantiles(bins_for_quantiles=bins_for_quantiles)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])
    rqf.setBackground(f"{dataDir}/bkg_para_postfit.pkl", bhist_bkg, sf=1.0)

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False, ext=f"{dataDir}/zmumu_para_postfit.pkl")
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False, xMin=-100, xMax=100)
    rqf.plot_refit_quantiles(prefit=False, logY=False, xMin=-100, xMax=100)
    rqf.qparms_postfit()


def zmumu_perp_gen():
    proc = "zmumu_gen"
    plotDir = "%s/%s_perp" % (basePlotDir, proc)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus} (q_{T} gen)"
    metLabel = "%s, U_{#perp}" % (met)
    utils.mkdir(plotDir, False)
    bhist = data['zmumu_perp_gen']

    recoilRebin = [-100, -70, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 70, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 1)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "perp", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_perp", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=recoilRebin)
    rqf.setQuantiles(bins_for_quantiles=recoilRebin)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False) # ext=f"data/zmumu_perp_{met}_postfit.pkl"
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()

    rqf.plot_refit_quantiles(ext=f"{dataDir}/zmumu_perp_postfit.pkl")


def zmumu_para_gen():
    proc = "zmumu_gen"
    plotDir = "%s/%s_para" % (basePlotDir, proc)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus} (q_{T} gen)"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(plotDir, False)
    bhist = data['zmumu_para_gen']

    recoilRebin = [-100, -70, -50, -45, -40, -35, -30] + list(range(-25, 25, 2)) + [25, 30, 35, 40, 45, 50, 55, 60, 70, 100]
    qTRebin = list(range(0, 50, 1)) + list(range(50, 100, 1)) + [100]

    rqf = qf.QuantileFitter(proc, met, procLabel, metLabel, dataDir, plotDir, "para", lumiLabel, logging)
    rqf.setHistConfig(bhist, "recoil_para", min(qTRebin), max(qTRebin), min(recoilRebin), max(recoilRebin), qTRebin=qTRebin, recoilRebinFit=1, recoilRebinPlt=recoilRebin)
    rqf.setQuantiles(bins_for_quantiles=recoilRebin)
    rqf.setSplineConfig([2, 6, 20, 60], extrpl=[None, 60])

    rqf.qparms_prefit()
    rqf.plot_refit_quantiles(prefit=True)
    parms = rqf.fit(withConstraint=False) # , ext=f"{dataDir}/zmumu_para_postfit.pkl"
    rqf.fit(withConstraint=True, parms=parms)
    rqf.plot_refit_quantiles(prefit=False)
    rqf.plot_refit_quantiles(prefit=False, logY=False)
    rqf.qparms_postfit()

    rqf.plot_refit_quantiles(ext=f"{dataDir}/zmumu_para_postfit.pkl")


def response_para():
    tag = "singlemuon_para_response"
    plotDir = "%s/%s" % (basePlotDir, tag)
    proc = "singlemuon"
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    utils.mkdir(plotDir, False)
    bhist_data = data['singlemuon_para']
    bhist_bkg = data['bkg_para']
    bhist = bhist_data + bhist_bkg*(-1)

    qTrebin = list(range(0, 100, 1)) + [100]
    corr = rc.ResponseCorr(proc, met, procLabel, metLabel, dataDir, plotDir, lumiLabel)
    corr.setHist(bhist, qTrebin)
    corr.correct(knots=[0, 2, 4, 6, 10, 15, 20, 30, 40, 50, 100], extrpl=[None, 50])



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    met = "DeepMETReso"
    flavor = "mumu"
    lumiLabel = "16.8 fb^{#minus1} (13 TeV)"

    with open("data/recoil_RawPFMET_highPU.pkl", "rb") as f:
        data = pickle.load(f)

    basePlotDir = "/home/submit/jaeyserm/public_html/recoil/highPU/RawPFMET_wmass/"
    dataDir = "data/highPU_RawPFMET/"
    utils.mkdir(basePlotDir, False)
    utils.mkdir(dataDir, False)

    #zmumu_para()
    zmumu_perp()

    #bkg_para()
    #bkg_perp()

    #singlemuon_para()
    #singlemuon_perp()
    
    #zmumu_perp_gen()
    #zmumu_para_gen()

    #response_para()