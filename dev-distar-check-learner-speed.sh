export cache=/mnt/cache/$USER
cd $cache
cd DI-star-of-mine
git checkout dev-distar-check-learner-speed
/mnt/lustre/share/git pull
source /mnt/cache/share/spring/s0.3.4
export SC2PATH="/mnt/lustre/zhumengshen.vendor/StarCraftII_4.10.0"
srun --partition=cpu --job-name=league_learner_speed -w $1 --mpi=pmi2 --async --output=./check-learner-speed-logs/league_output_check_learner_speed.txt python -m distar.bin.rl_train --type league --task selfplay --coordinator_ip $2
srun --partition=cpu --job-name=coordinator_learner_speed -w $1 --mpi=pmi2 --async --output=./check-learner-speed-logs/coordinator_output_check_learner_speed.txt python -m distar.bin.rl_train --type coordinator --coordinator_ip $2
srun --partition=GAME -N 2 --gres=gpu:8 --ntasks-per-node 8 --job-name=learner_learner_speed -w SH-IDC1-10-198-34-[142,146] --mpi=pmi2 --async --output=./check-learner-speed-logs/learner_output_check_learner_speed.txt python -m distar.bin.rl_train --type learner --coordinator_ip $2