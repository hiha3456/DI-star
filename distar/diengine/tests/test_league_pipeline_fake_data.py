import logging
import pytest
from easydict import EasyDict
from copy import deepcopy
from unittest.mock import patch
from typing import TYPE_CHECKING, Any, List, Dict, Optional

from ding.data.buffer.middleware import use_time_check
from ding.data import DequeBuffer
from ding.envs import EnvSupervisor
from ding.framework.supervisor import ChildType
from ding.framework.context import BattleContext
from ding.framework.middleware import StepLeagueActor, LeagueCoordinator, LeagueLearnerCommunicator, data_pusher, OffPolicyLearner
from ding.framework.task import task, Parallel
from ding.league.v2 import BaseLeague
from distar.diengine.config import distar_cfg
from distar.diengine.envs.distar_env import DIStarEnv
from distar.diengine.envs.fake_data import rl_step_data
from distar.diengine.policy.distar_policy import DIStarPolicy
from distar.diengine.middleware import DIstarBattleStepCollector, last_step_fn


class DIstarCollectMode:

    def __init__(self) -> None:
        self._cfg = EasyDict(dict(collect=dict(n_episode=1)))
        self._race = 'zerg'

    def load_state_dict(self, state_dict):
        return

    def get_attribute(self, name: str) -> Any:
        if hasattr(self, '_get_' + name):
            return getattr(self, '_get_' + name)()
        elif hasattr(self, '_' + name):
            return getattr(self, '_' + name)
        else:
            raise NotImplementedError

    def reset(self, data_id: Optional[List[int]] = None) -> None:
        pass

    def forward(self, policy_obs: Dict[int, Any]) -> Dict[int, Any]:
        # print("Call forward_collect:")
        return_data = {}
        return_data['action'] = DIStarEnv.random_action(policy_obs)
        return_data['logit'] = [1]
        return_data['value'] = [0]

        return return_data

    def process_transition(self, obs, model_output, timestep) -> dict:
        step_data = rl_step_data()
        step_data['done'] = timestep.done
        return step_data


class DIStarMockPolicyCollect:

    def __init__(self):

        self.collect_mode = DIstarCollectMode()


env_cfg = dict(
    actor=dict(job_type='train', ),
    env=dict(
        map_name='KingsCove',
        player_ids=['agent1', 'agent2'],
        races=['zerg', 'zerg'],
        map_size_resolutions=[True, True],  # if True, ignore minimap_resolutions
        minimap_resolutions=[[160, 152], [160, 152]],
        realtime=False,
        replay_dir='.',
        random_seed='none',
        game_steps_per_episode=100000,
        update_bot_obs=False,
        save_replay_episodes=1,
        update_both_obs=False,
        version='4.10.0',
    ),
)
env_cfg = EasyDict(env_cfg)
cfg = deepcopy(distar_cfg)


class PrepareTest():

    @classmethod
    def get_env_fn(cls):
        return DIStarEnv(env_cfg)

    @classmethod
    def get_env_supervisor(cls):
        for _ in range(10):
            try:
                env = EnvSupervisor(
                    type_=ChildType.THREAD,
                    env_fn=[cls.get_env_fn for _ in range(cfg.env.collector_env_num)],
                    **cfg.env.manager
                )
                env.seed(cfg.seed)
                return env
            except Exception as e:
                print(e)
                continue

    @classmethod
    def policy_fn(cls):
        policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['learn'])
        return policy

    @classmethod
    def collect_policy_fn(cls):
        policy = DIStarMockPolicyCollect()
        return policy


def main():
    logging.getLogger().setLevel(logging.INFO)
    league = BaseLeague(cfg.policy.other.league)
    N_PLAYERS = len(league.active_players_ids)
    print("League: n_players =", N_PLAYERS)

    with task.start(async_mode=True, ctx=BattleContext()),\
      patch("ding.framework.middleware.league_actor.BattleStepCollector", DIstarBattleStepCollector):
        print("node id:", task.router.node_id)
        if task.router.node_id == 0:
            coordinator_league = BaseLeague(cfg.policy.other.league)
            task.use(LeagueCoordinator(cfg, coordinator_league))
        elif task.router.node_id <= N_PLAYERS:
            cfg.policy.collect.unroll_len = 1
            player = league.active_players[task.router.node_id % N_PLAYERS]

            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            buffer_.use(use_time_check(buffer_, max_use=cfg.policy.other.replay_buffer.max_use))
            policy = PrepareTest.policy_fn()

            task.use(LeagueLearnerCommunicator(cfg, policy.learn_mode, player))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        else:
            task.use(StepLeagueActor(cfg, PrepareTest.get_env_supervisor, PrepareTest.collect_policy_fn, last_step_fn))

        task.run()


@pytest.mark.unittest
def test_league_pipeline():
    Parallel.runner(n_parallel_workers=4, protocol="tcp", topology="mesh")(main)


if __name__ == "__main__":
    Parallel.runner(n_parallel_workers=4, protocol="tcp", topology="mesh")(main)
