#!/bin/bash

cd $ZOE_WORKSPACE/code1.0

pip install sklearn

echo 'RANDOM'
PYTHONPATH=. python ./experiments/dgp_pca_oil.py --q_Omega_fixed=2500 --theta_fixed=7500 --is_ard=False --optimizer=adam --nl=1 --learning_rate=0.01 --n_rff=100 --df=3 --mc_train=50 --mc_test=10 --n_iterations=50000 --display_step=100 --duration=60 --learn_Omega=optim --less_prints=False --initializer=RANDOM > ../RANDOM_console.txt

echo 'PCA'
PYTHONPATH=. python ./experiments/dgp_pca_oil.py --q_Omega_fixed=2500 --theta_fixed=7500 --is_ard=False --optimizer=adam --nl=1 --learning_rate=0.01 --n_rff=100 --df=3 --mc_train=50 --mc_test=10 --n_iterations=50000 --display_step=100 --duration=60 --learn_Omega=optim --less_prints=False --initializer=PCA > ../PCA_console.txt

echo 'KernelPCA'
PYTHONPATH=. python ./experiments/dgp_pca_oil.py --q_Omega_fixed=2500 --theta_fixed=7500 --is_ard=False --optimizer=adam --nl=1 --learning_rate=0.01 --n_rff=100 --df=3 --mc_train=50 --mc_test=10 --n_iterations=50000 --display_step=100 --duration=60 --learn_Omega=optim --less_prints=False --initializer=KernelPCA > ../KernelPCA_console.txt

echo 'ISOMAP'
PYTHONPATH=. python ./experiments/dgp_pca_oil.py --q_Omega_fixed=2500 --theta_fixed=7500 --is_ard=False --optimizer=adam --nl=1 --learning_rate=0.01 --n_rff=100 --df=3 --mc_train=50 --mc_test=10 --n_iterations=50000 --display_step=100 --duration=60 --learn_Omega=optim --less_prints=False --initializer=ISOMAP > ../ISOMAP_console.txt

