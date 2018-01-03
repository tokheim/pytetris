#!/usr/bin/env bash
source env/bin/activate
python -m tensorboard.main --logdir=summary
