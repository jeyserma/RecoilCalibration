# RecoilCalibration

Clone the repo:

```shell
git clone --recurse-submodules git@github.com:jeyserma/RecoilCalibration.git
```


Need to run in singularity:

```shell
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:latest
source setup.sh
```

Recoil fits (for parallel/perpendicular data/mc/bkg):

```shell
python calibration/highPU_DeepMETReso.py
```


Export to TFLite model

```shell
python python/export.py
```

