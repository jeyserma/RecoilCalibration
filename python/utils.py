
import sys,array,math,os,copy,shutil,decimal
import json

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


def getNonZeroMinimum(h):
    yMin = 9e99
    for i in range(1, h.GetNbinsX()+1):
        if h.GetBinContent(i) < yMin and h.GetBinContent(i) > 1e-20:
            yMin= h.GetBinContent(i)
    return yMin
