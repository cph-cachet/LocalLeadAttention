#!/bin/bash
set -x
python train_model.py training_data model
python test_model.py model test_data test_outputs
python evaluate_model.py test_data test_outputs
