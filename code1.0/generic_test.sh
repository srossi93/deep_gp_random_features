PYTHONPATH=. python ./experiments/dgp_clustering.py \
                    --clustering=True --q_Omega_fixed=2500 --theta_fixed=7500 --is_ard=False    \
                    --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100  --mc_train=1  \
                    --mc_test=1 --n_iterations=20000 --display_step=1000 --duration=60 \
                    --learn_Omega=optim --less_prints=True --initializer=RANDOM --df=2 --seed=0
cd img/
mkdir -p seed0
for i in *.pdf; do mv "$i" seed0/"$i"; done
cd ../



PYTHONPATH=. python ./experiments/dgp_clustering.py \
                    --clustering=True --q_Omega_fixed=2500 --theta_fixed=7500 --is_ard=False    \
                    --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100  --mc_train=1  \
                    --mc_test=1 --n_iterations=20000 --display_step=1000 --duration=60 \
                    --learn_Omega=optim --less_prints=True --initializer=RANDOM --df=2 --seed=10
cd img/
mkdir -p seed10
for i in *.pdf; do mv "$i" seed10/"$i"; done
cd ../

PYTHONPATH=. python ./experiments/dgp_clustering.py \
                    --clustering=True --q_Omega_fixed=2500 --theta_fixed=7500 --is_ard=False    \
                    --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100  --mc_train=1  \
                    --mc_test=1 --n_iterations=20000 --display_step=1000 --duration=60 \
                    --learn_Omega=optim --less_prints=True --initializer=RANDOM --df=2 -seed=100
cd img/
mkdir -p seed100
for i in *.pdf; do mv "$i" seed100/"$i"; done
cd ../

PYTHONPATH=. python ./experiments/dgp_clustering.py \
                    --clustering=True --q_Omega_fixed=2500 --theta_fixed=7500 --is_ard=False    \
                    --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100  --mc_train=1  \
                    --mc_test=1 --n_iterations=20000 --display_step=1000 --duration=60 \
                    --learn_Omega=optim --less_prints=True --initializer=RANDOM --df=2 --seed=1000
cd img/
mkdir -p seed1000
for i in *.pdf; do mv "$i" seed1000/"$i"; done
cd ../

PYTHONPATH=. python ./experiments/dgp_clustering.py \
                    --clustering=True --q_Omega_fixed=2500 --theta_fixed=7500 --is_ard=False    \
                    --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100  --mc_train=1  \
                    --mc_test=1 --n_iterations=20000 --display_step=1000 --duration=60 \
                    --learn_Omega=optim --less_prints=True --initializer=RANDOM --df=2 --seed=12837
cd img/
mkdir -p seed12837
for i in *.pdf; do mv "$i" seed12837/"$i"; done
cd ../
