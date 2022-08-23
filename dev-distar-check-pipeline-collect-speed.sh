export cache=/mnt/cache/$USER
cd $cache
cd DI-star-of-mine
git checkout dev-distar-check-pipeline-collect-speed
/mnt/lustre/share/git pull
source /mnt/cache/share/spring/s0.3.4
export SC2PATH="/mnt/lustre/zhumengshen.vendor/StarCraftII_4.10.0"
srun --partition=cpu --quotatype spot --job-name=league_pipeline_collect_speed_$6 -w $1 \
    --mpi=pmi2 --async --output=./check-pipeline-collect-speed-logs-$6/league_output_check_pipeline_collect_speed_$6.txt \
    python -m distar.bin.rl_train --type league --task selfplay --coordinator_ip $2
srun --partition=cpu --quotatype spot --job-name=coordinator_pipeline_collect_speed_$6 -w $1 \
    --mpi=pmi2 --async --output=./check-pipeline-collect-speed-logs-$6/coordinator_output_check_pipeline_collect_speed_$6.txt \
    python -m distar.bin.rl_train --type coordinator --coordinator_ip $2
srun --partition=cpu --quotatype spot --job-name=actor_pipeline_collect_speed_$6 -w $1 -c $7 \
    --mpi=pmi2 --async --output=./check-pipeline-collect-speed-logs-$6/actor_output_check_pipeline_collect_speed_$6.txt \
    python -m distar.bin.rl_train --type actor --coordinator_ip $2 --env_num $6
for ((i=0; i<8; i++))
do
    srun --partition=GAME -w $3 --quotatype spot --gres=gpu:1 --ntasks-per-node 1 -c 32 \
        --job-name=learner_pipeline_collect_speed_$6 --mpi=pmi2 --async \
        --output=./check-pipeline-collect-speed-logs-$6/learner_output_check_pipeline_collect_speed_$6_$i.txt \
        python -m distar.bin.rl_train --type learner --coordinator_ip $2 --init_method $5 \
        --rank $i --world_size 16
done
for ((i=8; i<16; i++))
do
    srun --partition=GAME -w $4 --quotatype spot --gres=gpu:1 --ntasks-per-node 1 -c 32 \
        --job-name=learner_pipeline_collect_speed_$6 --mpi=pmi2 --async \
        --output=./check-pipeline-collect-speed-logs-$6/learner_output_check_pipeline_collect_speed_$6_$i.txt \
        python -m distar.bin.rl_train --type learner --coordinator_ip $2 --init_method $5 \
        --rank $i --world_size 16
done