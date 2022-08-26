from easydict import EasyDict
from typing import Dict, TYPE_CHECKING, List, Callable
import time
from ditk import logging

from ding.envs import BaseEnvManager, BaseEnvTimestep
from ding.utils import log_every_sec
from ding.framework import task
from ding.framework.middleware.functional import PlayerModelInfo
from ding.framework.middleware.functional.collector import BattleTransitionList
import treetensor.torch as ttorch

if TYPE_CHECKING:
    from ding.framework import BattleContext


def battle_inferencer_for_distar(cfg: EasyDict, env: BaseEnvManager):

    def _battle_inferencer(ctx: "BattleContext"):
        # Get current env obs.
        obs = env.ready_obs
        assert isinstance(obs, dict)

        ctx.obs = obs

        # Policy forward.
        inference_output = {}
        actions = {}
        for env_id in ctx.obs.keys():
            observations = obs[env_id]
            inference_output[env_id] = {}
            actions[env_id] = {}
            for policy_id, policy_obs in observations.items():
                # policy.forward
                output = ctx.current_policies[policy_id].forward(policy_obs)
                inference_output[env_id][policy_id] = output
                actions[env_id][policy_id] = output['action']
        ctx.inference_output = inference_output
        ctx.actions = actions

    return _battle_inferencer


def battle_rolloutor_for_distar(cfg: EasyDict, env: BaseEnvManager, transitions_list: List, model_info_dict: Dict):

    def _battle_rolloutor(ctx: "BattleContext"):
        timesteps = env.step(ctx.actions)
        ctx.total_envstep_count += len(timesteps)
        ctx.env_step += len(timesteps)

        if isinstance(timesteps, list):
            new_time_steps = {}
            for env_id, timestep in enumerate(timesteps):
                new_time_steps[env_id] = timestep
            timesteps = new_time_steps

        for env_id, timestep in timesteps.items():
            if timestep.info.get('abnormal'):
                for policy_id, policy in enumerate(ctx.current_policies):
                    transitions_list[policy_id].clear_newest_episode(env_id, before_append=True)
                    policy.reset(env.ready_obs[0][policy_id])
                continue

            episode_long_enough = True
            for policy_id, policy in enumerate(ctx.current_policies):
                if timestep.obs.get(policy_id):
                    policy_timestep = BaseEnvTimestep(
                        obs=timestep.obs.get(policy_id),
                        reward=timestep.reward[policy_id],
                        done=timestep.done,
                        info=timestep.info[policy_id]
                    )
                    transition = policy.process_transition(obs=None, model_output=None, timestep=policy_timestep)
                    transition = EasyDict(transition)
                    transition.collect_train_iter = ttorch.as_tensor(
                        [model_info_dict[ctx.player_id_list[policy_id]].update_train_iter]
                    )

                    # 2nd case when the number of transitions in one of all the episodes is shorter than unroll_len
                    episode_long_enough = episode_long_enough and transitions_list[policy_id].append(env_id, transition)

            if timestep.done:
                for policy_id, policy in enumerate(ctx.current_policies):
                    policy.reset(env.ready_obs[0][policy_id])
                    ctx.episode_info[policy_id].append(timestep.info[policy_id])

            if not episode_long_enough:
                for policy_id, _ in enumerate(ctx.current_policies):
                    transitions_list[policy_id].clear_newest_episode(env_id)
                    ctx.episode_info[policy_id].pop()
            elif timestep.done:
                ctx.env_episode += 1

    return _battle_rolloutor


WAIT_MODEL_TIME = float('inf')


def last_step_fn(last_step):
    for k in ['mask', 'action_info', 'teacher_logit', 'behaviour_logp', 'selected_units_num', 'reward', 'step']:
        last_step.pop(k)
    return last_step



class DIstarBattleStepCollector:

    def __init__(
        self, cfg: EasyDict, env: BaseEnvManager, unroll_len: int, model_dict: Dict, model_info_dict: Dict,
        player_policy_collect_dict: Dict, agent_num: int, last_step_fn: Callable = None
    ):
        self.cfg = cfg
        self.end_flag = False
        # self._reset(env)
        self.env = env
        self.env_num = self.env.env_num

        self.total_envstep_count = 0
        self.unroll_len = unroll_len
        self.model_dict = model_dict
        self.model_info_dict = model_info_dict
        self.player_policy_collect_dict = player_policy_collect_dict
        self.agent_num = agent_num

        self._battle_inferencer = task.wrap(battle_inferencer_for_distar(self.cfg, self.env))
        self._transitions_list = [
            BattleTransitionList(self.env.env_num, self.unroll_len, last_step_fn) for _ in range(self.agent_num)
        ]
        self._battle_rolloutor = task.wrap(
            battle_rolloutor_for_distar(self.cfg, self.env, self._transitions_list, self.model_info_dict)
        )

    def __del__(self) -> None:
        """
        Overview:
            Execute the close command and close the collector. __del__ is automatically called to \
                destroy the collector instance when the collector finishes its work
        """
        if self.end_flag:
            return
        self.end_flag = True
        self.env.close()

    def _update_policies(self, player_id_set) -> None:
        for player_id in player_id_set:
            # for this player, if in the beginning of actor's lifetime, 
            # actor didn't recieve any new model, use initial model instead.
            if self.model_info_dict.get(player_id) is None:
                self.model_info_dict[player_id] = PlayerModelInfo(
                    get_new_model_time=time.time(), update_new_model_time=None
                )

        update_player_id_set = set()
        for player_id in player_id_set:
            if 'historical' not in player_id:
                update_player_id_set.add(player_id)
        while True:
            time_now = time.time()
            time_list = [time_now - self.model_info_dict[player_id].get_new_model_time for player_id in update_player_id_set]
            if any(x >= WAIT_MODEL_TIME for x in time_list):
                for index, player_id in enumerate(update_player_id_set):
                    if time_list[index] >= WAIT_MODEL_TIME:
                        log_every_sec(
                            logging.WARNING, 5,
                            'In actor {}, model for {} is not updated for {} senconds, and need new model'.format(
                                task.router.node_id, player_id, time_list[index]
                            )
                        )
                time.sleep(1)
            else:
                break

        for player_id in update_player_id_set:
            if self.model_dict.get(player_id) is None:
                continue
            else:
                learner_model = self.model_dict.get(player_id)
                policy = self.player_policy_collect_dict.get(player_id)
                assert policy, "for player{}, policy should have been initialized already"
                # update policy model
                policy.load_state_dict(learner_model.state_dict)
                self.model_info_dict[player_id].update_new_model_time = time.time()
                self.model_info_dict[player_id].update_train_iter = learner_model.train_iter
                self.model_dict[player_id] = None

    def __call__(self, ctx: "BattleContext") -> None:

        ctx.total_envstep_count = self.total_envstep_count
        old = ctx.env_step

        while True:
            if self.env.closed:
                self.env.launch()
                for policy_id, policy in enumerate(ctx.current_policies):
                    policy.reset(self.env.ready_obs[0][policy_id])
            self._update_policies(set(ctx.player_id_list))
            try:
                self._battle_inferencer(ctx)
                self._battle_rolloutor(ctx)
            except Exception as e:
                logging.error("[Actor {}] got an exception: {} when collect data".format(task.router.node_id, e))
                self.env.close()
                for env_id in range(self.env_num):
                    for policy_id, policy in enumerate(ctx.current_policies):
                        self._transitions_list[policy_id].clear_newest_episode(env_id, before_append=True)

            self.total_envstep_count = ctx.total_envstep_count

            only_finished = True if ctx.env_episode >= ctx.n_episode else False
            if (self.unroll_len > 0 and ctx.env_step - old >= self.unroll_len) or ctx.env_episode >= ctx.n_episode:
                for transitions in self._transitions_list:
                    trajectories = transitions.to_trajectories(only_finished=only_finished)
                    ctx.trajectories_list.append(trajectories)
                if ctx.env_episode >= ctx.n_episode:
                    self.env.close()
                    ctx.job_finish = True
                    for transitions in self._transitions_list:
                        transitions.clear()
                break