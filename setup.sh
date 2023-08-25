export REC_BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

source ${REC_BASE}/narf/setup.sh

export PYTHONPATH="${REC_BASE}:$PYTHONPATH"
export PYTHONPATH="${REC_BASE}/python:$PYTHONPATH"


export TF_ENABLE_ONEDNN_OPTS="1"
export TF_NUM_INTEROP_THREADS="32"
export TF_NUM_INTRAOP_THREADS="32"