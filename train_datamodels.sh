#!/bin/bash
#SBATCH --job-name=fit_datamodels 
#SBATCH --time=15:00:00              
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --output=./bash_out/%x_%A_%a.out
echo "Checking if datamodels_input.beton exists..."

if [ ! -f "./fit_datamodels/updated_alpa01_59k/datamodels_input.beton" ]; then
    python write_dataset_regression_with_concatenation.py --config.input_directory='./train_CIFAR10_out/temp' \
                                                    --config.output_directory='./fit_datamodels/updated_alpa01_59k' \
                                                    --config.x_name='masks' \
                                                    --config.y_name='margins' \
                                                    --config.prefix='trial' \
                                                    --config.nmodels=60000 \
                                                    --config.length=25000 \
                                                    --config.val_length=10000
else
    echo "exists."
fi

echo "Computing datamodels..."

python compute_datamodels.py \
    --data.data_path='./fit_datamodels/updated_alpa01_59k/datamodels_input.beton' \
    --data.num_train=50000 \
    --data.num_val=8000 \
    --data.target_start_ind=0 \
    --data.target_end_ind=-1 \
    --cfg.out_dir='./fit_datamodels/updated_alpa01_59k/' \
    --cfg.num_workers=1 \
    --cfg.batch_size=2000 \
    --cfg.k=100 \
    --cfg.lr=5e-3 \
    --cfg.eps=1e-6 \
    --early_stopping.check_every=3 \
    --early_stopping.eps=5e-11
