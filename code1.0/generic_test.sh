for iter in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    export PYTHONPATH=.
    echo Run experiment iter = $iter
    python experiments/dgp_pca_oil.py  --theta_fixed=29000 \
                                              --is_ard=False --optimizer=adam --nl=1 \
                                              --learning_rate=0.01 --n_rff=50 --df=3 \
                                              --mc_train=1 --mc_test=1 --n_iterations=$iter \
                                              --display_step=1 --duration=60 \
                                              --learn_Omega=no --less_prints=False 
    mv latent_space.pdf iter${iter}.pdf
done
