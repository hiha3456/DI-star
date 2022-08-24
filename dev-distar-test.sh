export cache=/mnt/cache/$USER
cd $cache
cd DI-star
git checkout change_config
/mnt/lustre/share/git pull
source /mnt/cache/share/spring/s0.3.4
export SC2PATH="/mnt/lustre/zhumengshen.vendor/StarCraftII_4.10.0"
srun --partition=cpu --quotatype spot -w $1 \
    --mpi=pmi2 --async \
    python -m distar.bin.rl_train --type league --task selfplay --coordinator_ip $2
srun --partition=cpu --quotatype spot -w $1 \
    --mpi=pmi2 --async \
    python -m distar.bin.rl_train --type coordinator --coordinator_ip $2
srun --partition=cpu --quotatype spot -n $6 -c 15 \
    --mpi=pmi2 --async \
    python -m distar.bin.rl_train --type actor --coordinator_ip $2 --env_num 3
for ((i=0; i<8; i++))
do
    srun --partition=GAME -w $3 --quotatype spot --gres=gpu:1 --ntasks-per-node 1 -c 32 \
        --mpi=pmi2 --async \
        python -m distar.bin.rl_train --type learner --coordinator_ip $2 --init_method $5 \
        --rank $i --world_size 16
done
for ((i=8; i<16; i++))
do
    srun --partition=GAME -w $4 --quotatype spot --gres=gpu:1 --ntasks-per-node 1 -c 32 \
        --mpi=pmi2 --async \
        python -m distar.bin.rl_train --type learner --coordinator_ip $2 --init_method $5 \
        --rank $i --world_size 16
done