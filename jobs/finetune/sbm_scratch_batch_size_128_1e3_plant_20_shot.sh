#!/bin/bash
#SBATCH --mail-user=ar.aamer@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=20shot_plant_sbm_scratch_batch_size_512_adam_1e3_finetune
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=2-00:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

module load python
source ~/py37/bin/activate


echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/proj_cdf2 .

echo "Copying the datasets"
date +"%T"
cp -r ~/CDFSL_Datasets .


echo "creating data directories"
date +"%T"
cd proj_cdf2
cd data
unzip -q $SLURM_TMPDIR/CDFSL_Datasets/miniImagenet.zip

mkdir Plant_disease 

cd Plant_disease
unzip -q $SLURM_TMPDIR/CDFSL_Datasets/plant-disease.zip
cd ..

cd $SLURM_TMPDIR
cd proj_cdf2



echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR

cd proj_cdf2


echo "**********************20-shot-plant-sbm_scratch_batch_size_512_adam_1e3**************************"
python sbm_scratch_batch_size_512_adam_1e3_plant_finetune.py --model ResNet10 --method baseline --n_shot 20 --freeze_backbone




wait


echo "---------------------------------------<End of program>-------------------------------------"
date +"%T"



