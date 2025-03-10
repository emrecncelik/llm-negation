#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Usage: $0 <model_type>"
  exit 1
fi

model_type="$1"
configs=($(ls -1 configs | sort))

for cfg in "${configs[@]}"; do
  if [[ $cfg == $model_type* ]]; then
    echo "Current config: $cfg"
    python run_experiment.py --config configs/$cfg
  fi
done