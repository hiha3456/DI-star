export cache=/mnt/cache/$USER
cd $cache
cd DI-star-of-mine
git checkout dev-distar-check-pipeline-collect-speed
/mnt/lustre/share/git pull
source /mnt/cache/share/spring/s0.3.4
export SC2PATH="/mnt/lustre/zhumengshen.vendor/StarCraftII_4.10.0"
srun --partition=cpu --quotatype spot --job-name=league_pipeline_collect_speed_$4 -w $1 \
    --mpi=pmi2 --async --output=./check-pipeline-collect-speed-logs-$4/league_output_check_pipeline_collect_speed_$4.txt \
    python -m distar.bin.rl_train --type league --task selfplay --coordinator_ip $2
srun --partition=cpu --quotatype spot --job-name=coordinator_pipeline_collect_speed_$4 -w $1 \
    --mpi=pmi2 --async --output=./check-pipeline-collect-speed-logs-$4/coordinator_output_check_pipeline_collect_speed_$4.txt \
    python -m distar.bin.rl_train --type coordinator --coordinator_ip $2
srun --partition=cpu --quotatype spot -n $4 --job-name=actor_pipeline_collect_speed_$4 -c 15 \
    --mpi=pmi2 --async --output=./check-pipeline-collect-speed-logs-$4/actor_output_check_pipeline_collect_speed_$4.txt \
    python -m distar.bin.rl_train --type actor --coordinator_ip $2 --env_num 3
srun --partition=GAME --quotatype spot -w $3 -n 16 --gres=gpu:8 --ntasks-per-node 8 -c 32 \
    --job-name=learner_pipeline_collect_speed_$4 --mpi=pmi2 --async \
     --output=./check-pipeline-collect-speed-logs-$4/learner_output_check_pipeline_collect_speed_$4.txt \
    python -m distar.bin.rl_train --type learner --coordinator_ip $2