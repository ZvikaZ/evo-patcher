#!/bin/bash

#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --job-name patcher		### name of the job
#SBATCH --gpus=1				### number of GPUs, allocating more than 1 requires IT team's permission

##SBATCH --mail-user=zvikah@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=END,FAIL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

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
