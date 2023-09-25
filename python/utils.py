
import sys,array,math,os,copy,shutil,decimal
import json
import ROOT
import random

def mkdir(outDir, remove=True):
    if os.path.exists(outDir) and os.path.isdir(outDir) and remove: shutil.rmtree(outDir)
    os.system("mkdir -p %s" % outDir)


def drange(x, y, jump):
    while x < y:
        yield float(x)
        #x += decimal.Decimal(jump)
        x += jump


def loadJSON(jsIn):
    with open(jsIn) as f: jsDict = json.load(f)
    return jsDict

def writeJSON(jsOut, outDict):
    with open(jsOut, "w") as outfile: json.dump(outDict, outfile, indent=4)


def getNonZeroMinimum(h, xMin=-9e99, xMax=9e99):
    yMin = 9e99
    for i in range(1, h.GetNbinsX()+1):
        if h.GetBinCenter(i) < xMin or h.GetBinCenter(i) > xMax:
            continue
        if h.GetBinContent(i) < yMin and h.GetBinContent(i) > 1e-20:
            yMin= h.GetBinContent(i)
    return yMin


def makeTGraph(x, y, x_err=[], y_err=[], style="marker"):
    gr = ROOT.TGraphErrors()
    gr.SetName(f"{random.randint(0,9999)}")
    if style == "marker":
        gr.SetLineColor(ROOT.kBlack)
        gr.SetMarkerStyle(20)
        gr.SetMarkerSize(0.3)
        gr.SetMarkerColor(ROOT.kBlack)
        gr.SetLineColor(ROOT.kBlack)
    else:
        gr.SetLineWidth(2)
        gr.SetMarkerSize(0)
        gr.SetLineColor(ROOT.kRed)
    if len(x_err) == 0:
        x_err = [0]*len(x)
    if len(y_err) == 0:
        y_err = [0]*len(x)
    for i in range(0, len(x)):
        gr.SetPoint(i, x[i], y[i])
        gr.SetPointError(i, x_err[i], y_err[i])
    return gr