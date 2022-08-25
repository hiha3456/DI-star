export cache=/mnt/cache/$USER
cd $cache
cd DI-star-of-mine
git checkout dev-distar-check-learner-speed-torch
/mnt/lustre/share/git pull
source /mnt/cache/share/spring/s0.3.4
export SC2PATH="/mnt/lustre/zhumengshen.vendor/StarCraftII_4.10.0"
srun --partition=cpu --quotatype spot --job-name=league_learner_speed -w $1 \
    --mpi=pmi2 --async --output=./check-learner-speed-logs/league_output_check_learner_speed.txt \
    python -m distar.bin.rl_train --type league --task selfplay --coordinator_ip $2
srun --partition=cpu --quotatype spot --job-name=coordinator_learner_speed -w $1 \
    --mpi=pmi2 --async --output=./check-learner-speed-logs/coordinator_output_check_learner_speed.txt \
    python -m distar.bin.rl_train --type coordinator --coordinator_ip $2
srun --partition=GAME --quotatype spot -w $3 -n 16 --gres=gpu:8 --ntasks-per-node 8 -c 32 \
    --job-name=learner_learner_speed --mpi=pmi2 --async \
     --output=./check-learner-speed-logs/learner_output_check_learner_speed.txt\
    python -m distar.bin.rl_train --type learner --coordinator_ip $2