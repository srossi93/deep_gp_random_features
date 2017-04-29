#!/bin/bash

cd $ZOE_WORKSPACE/code1.0

pip install sklearn

echo 'Test...'
PYTHONPATH=. python ./experiments/dgp_pca_mnist.py --q_Omega_fixed=2500 --theta_fixed=7500 --is_ard=False --optimizer=adam --nl=1 --learning_rate=0.01 --n_rff=50 --df=3 --mc_train=1 --mc_test=1 --n_iterations=10 --display_step=1 --duration=60 --learn_Omega=optim --less_prints=False --initializer=RANDOM
echo 'Done!'

echo 'Starting training now..'
PYTHONPATH=. python ./experiments/dgp_pca_mnist.py --q_Omega_fixed=2500 --theta_fixed=7500 --is_ard=False --optimizer=adam --nl=1 --learning_rate=0.01 --n_rff=50 --df=3 --mc_train=1 --mc_test=1 --n_iterations=50000 --display_step=100 --duration=1440 --learn_Omega=optim --less_prints=False --initializer=PCA > ../mnist_console.txt
