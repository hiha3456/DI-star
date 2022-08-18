export cache=/mnt/cache/$USER
cd $cache
cd DI-star-of-mine
git checkout dev-distar-check-actor-num
/mnt/lustre/share/git pull
source /mnt/cache/share/spring/s0.3.4
export SC2PATH="/mnt/lustre/zhumengshen.vendor/StarCraftII_4.10.0"
srun --partition=cpu --job-name=league_actor_num_$1 -w $3 --mpi=pmi2 --async --output=./check-actor-num-logs/league_output_check_actor_num_$1.txt python -m distar.bin.rl_train --type league --task selfplay
srun --partition=cpu --job-name=coordinator_actor_num_$1 -w $3 --mpi=pmi2 --async --output=./check-actor-num-logs/coordinator_output_check_actor_num_$1.txt python -m distar.bin.rl_train --type coordinator
srun --partition=cpu --job-name=actor_actor_num_$1 -w $3 -c $2 --mpi=pmi2 --async --output=./check-actor-num-logs/actor_output_check_actor_num_$1.txt python -m distar.bin.rl_train --type actor --env_num $1
