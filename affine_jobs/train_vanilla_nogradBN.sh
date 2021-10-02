#!/bin/bash
#SBATCH --mail-user=ar.aamer@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=transfer_noBN_grad.sh
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-10:00
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




echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR

cd proj_cdf2
echo "********************************************************************************************"


python transfer_noBN2.py


wait

echo "---------------------------------------<Copying weights>------------------------------------"
cp -r logs/vanilla ~/proj_cdf2/logs

echo "---------------------------------------<End of program>-------------------------------------"
date +"%T"



