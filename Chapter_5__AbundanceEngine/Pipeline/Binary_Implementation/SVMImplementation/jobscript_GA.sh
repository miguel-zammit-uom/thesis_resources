#!/bin/bash
#
#SBATCH --job-name=GA_SVM_RUN11
#SBATCH --output=output_files/output_Run11.txt
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --nodelist=treachery
#

hostname 

source /users/mzammit/ML_Run_31_03_22/project_venv/bin/activate

/users/mzammit/ML_Run_31_03_22/project_venv/bin/python /users/mzammit/Abundance_clf_MOO_GA_Run_28_03_23/GA_with_SVM/Cluster_Pipeline/main.py