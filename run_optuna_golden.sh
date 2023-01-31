sbatch --wrap="python -u run_optuna.py" --job-name="optuna_g" --partition rtx3090 --qos sipper --gpus=rtx_3090:1
