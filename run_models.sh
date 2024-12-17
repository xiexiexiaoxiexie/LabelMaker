#!/bin/bash

WORKSPACE_DIR=/home/xiefujing/LabelMaker/azure/2024-11-03-09-47-53

python models/omnidata_depth.py --workspace $WORKSPACE_DIR
python models/hha_depth.py --workspace $WORKSPACE_DIR
python models/cmx.py --workspace $WORKSPACE_DIR

python models/mask3d_inst.py --workspace $WORKSPACE_DIR
python models/omnidata_normal.py --workspace $WORKSPACE_DIR
