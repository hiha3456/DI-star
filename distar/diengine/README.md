# Overview
A new pipeline of DI-star wrote by [DI-engine newest middleware](https://di-engine-docs.readthedocs.io/en/latest/03_system/middleware.html?highlight=middleware).

### Installation

#### 1. Install StarCraftII, distar and pytorch following [README of DI-star](https://github.com/opendilab/DI-star)

#### 2. Install DI-engine by
```bash
git clone https://github.com/opendilab/DI-engine.git
cd DI-engine
pip install -e .
```

#### 3. Install pytest by
```bash
pip install pytest
```

### Running pipeline on your machine
#### 1. download model sl_model in your **working directory**
```bash
python -m distar.bin.download_model --name sl_model
mv DI-star/distar/bin/sl_model.pth <your working directory>
```

#### 2. ajust model so it could be loaded by DI-star
```bash
python -m distar.diengine.state_dict_utils sl_model.pth
```

#### 3. run the pipeline by main entry or cmd tool ditask
[The readme of ditask](https://di-engine-docs.readthedocs.io/en/latest/03_system/distributed.html?highlight=ditask)

by main entry:
```bash
python distar/diengine/test/test_league_pipeline.py
```

by ditask
```bash
ditask --package distar.diengine.tests.test_league_pipeline --main distar.diengine.tests.test_league_pipeline.main --parallel-workers <number of parallel workers> --topology mesh
```

### Running pipeline on K8s

### Architecture of pipeline

#### Coordinator
The first node is coordinator, which combine the role of coordinator and league, to distribute job, recieve finished job to update payoff, recieve learner_meta to update players and create historical players

#### Learner
The next <number of main_player> nodes are learner, which used to train the main_players, and distribute models to actor, learner_meta to coordinator, and recieve trajectories from actor

#### Actor
The rest nodes of the pipeline are actors. They are used to collect data from SC2 envs, get job from coordinator, send datas to learners, and recieve models from them.

### Key Configurations
In DI-star/distar/diengine/config/distar_config.py
distar_cfg.policy.other.league.active_players.main_player: number of learner
you can also change replay_buffer_size, batch_size, unroll_len(traj_len) in it


