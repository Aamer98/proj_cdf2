#!/bin/bash
#SBATCH --mail-user=ar.aamer@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=sbm_scratch_batch_size_512_eurosat.sh
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=3-00:00
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

mkdir EuroSAT 

cd EuroSAT
unzip -q $SLURM_TMPDIR/CDFSL_Datasets/EuroSAT.zip
cd ..

cd $SLURM_TMPDIR
cd proj_cdf2
cd data
unzip -q $SLURM_TMPDIR/CDFSL_Datasets/eurosat_unlabelled_dataset.zip
unzip -q $SLURM_TMPDIR/CDFSL_Datasets/plant_disease_unlaballed.zip
cd ..


echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR

cd proj_cdf2
echo "********************************************************************************************"


python sbm_scratch_batch_size_512_adam_1e3_eurosat.py


wait

echo "---------------------------------------<Copying weights>------------------------------------"
cp -r logs/sbm_scratch_adam_1e3 ~/proj_cdf2/logs

echo "---------------------------------------<End of program>-------------------------------------"
date +"%T"



