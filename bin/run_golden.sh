#!/bin/bash

#SBATCH --partition rtx3090
#SBATCH --qos sipper
#SBATCH --gpus=rtx_3090:1

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  echo "Usage: $(basename $0) [--name run_name] [more options from main.py]"
  echo "Creates a new SLURM run, in regression/, with optional run_name added to dir"
  echo "Run 'python main.py -h' for all other options"
  echo ""
  exit 0
fi

if [ "$1" = "--name" ]; then
  run_dir=$(python misc.py --create-run-dir-all-runs regression --create-run-dir-run-name $2)
  shift
  shift
else
  run_dir=$(python misc.py --create-run-dir-all-runs regression)
fi

cd "$run_dir" || exit
cp ../../persist* .
cp ../../yolov5x.pt .
echo "Running in $run_dir"
echo python -u ../../main.py "$@"
python -u ../../main.py "$@"
