import os
import shutil
import socket
import subprocess
import time

import portpicker
import torch
from flask import Flask
from tensorboardX import SummaryWriter
from torch.optim.adam import Adam
import logging
from tensorboardX import SummaryWriter

from distar.ctools.utils import broadcast, EasyTimer
from distar.ctools.utils.config_helper import read_config, deep_merge_dicts, save_config
from distar.ctools.worker.learner.base_learner import BaseLearner
from distar.ctools.worker.learner.learner_comm import LearnerComm
from distar.ctools.worker.learner.learner_hook import LearnerHook, add_learner_hook
from .rl_training.rl_dataloader import RLDataLoader
from .rl_training.rl_loss import ReinforcementLoss
from .model.model import Model
import gc

PRE_TRAIN_ITER = 100

class MemCache:

    @staticmethod
    def byte2MB(bt):
        return round(bt / (1024 ** 2), 3)

    def __init__(self):
        self.dctn = {}
        self.max_reserved = 0
        self.max_allocate = 0

    def mclean(self):
        r0 = torch.cuda.memory_reserved(0)
        a0 = torch.cuda.memory_allocated(0)
        f0 = r0 - a0

        for key in list(self.dctn.keys()):
            del self.dctn[key]
        gc.collect()
        torch.cuda.empty_cache()

        r1 = torch.cuda.memory_reserved(0)
        a1 = torch.cuda.memory_allocated(0)
        f1 = r1 - a1

        print('Mem Free')
        print(f'Reserved  \t {MemCache.byte2MB(r1 - r0)}MB')
        print(f'Allocated \t {MemCache.byte2MB(a1 - a0)}MB')
        print(f'Free      \t {MemCache.byte2MB(f1 - f0)}MB')

    def __setitem__(self, key, value):
        self.dctn[key] = value
        self.max_reserved = max(self.max_reserved, torch.cuda.memory_reserved(0))
        self.max_allocate = max(self.max_allocate, torch.cuda.memory_allocated(0))

    def __getitem__(self, item):
        return self.dctn[item]

    def __delitem__(self, *keys):
        r0 = torch.cuda.memory_reserved(0)
        a0 = torch.cuda.memory_allocated(0)
        f0 = r0 - a0

        for key in keys:
            del self.dctn[key]

        r1 = torch.cuda.memory_reserved(0)
        a1 = torch.cuda.memory_allocated(0)
        f1 = r1 - a1

        print('Cuda Free')
        print(f'Reserved  \t {MemCache.byte2MB(r1 - r0)}MB')
        print(f'Allocated \t {MemCache.byte2MB(a1 - a0)}MB')
        print(f'Free      \t {MemCache.byte2MB(f1 - f0)}MB')

    def show_cuda_info(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a

        print('Cuda Info')
        print(f'Total     \t{MemCache.byte2MB(t)} MB')
        print(f'Reserved  \t{MemCache.byte2MB(r)} [{MemCache.byte2MB(self.max_reserved)}] MB')
        print(f'Allocated \t{MemCache.byte2MB(a)} [{MemCache.byte2MB(self.max_allocate)}] MB')
        print(f'Free      \t{MemCache.byte2MB(f)} MB')

class RLLearner(BaseLearner):
    def __init__(self, cfg, *args):
        self._job_type = cfg.learner.job_type
        super(RLLearner, self).__init__(cfg, *args)
        self._player_id = cfg.learner.player_id
        if self._job_type == 'train':
            self.comm = LearnerComm(cfg)
            add_learner_hook(self._hooks, SendModelHook(position='after_iter'))
            add_learner_hook(self._hooks, SendModelHook(position='before_run'))
            add_learner_hook(self._hooks, SendTrainInfo(position='after_iter'))
            self._ip = os.environ.get('SLURMD_NODENAME') if 'SLURMD_NODENAME' in os.environ else '127.0.0.1'
            self._port = portpicker.pick_unused_port()
            self._save_grad = cfg.learner.get('save_grad') and self.rank == 0
            if self._save_grad:
                self.grad_tb_path = os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                                 self.comm.player_id, 'grad')
                self.grad_tb_logger = SummaryWriter(self.grad_tb_path)
                self.clip_grad_tb_path = os.path.join(os.getcwd(), 'experiments',
                                                      self._whole_cfg.common.experiment_name,
                                                      self.comm.player_id, 'clip_grad')
                self.clip_grad_tb_logger = SummaryWriter(self.clip_grad_tb_path)
                self.model_tb_path = os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                                  self.comm.player_id, 'model')
                self.model_tb_logger = SummaryWriter(self.model_tb_path)
                self.save_log_freq = self._whole_cfg.learner.get('save_log_freq', 400)
            self._dataloader = RLDataLoader(cfg=self._whole_cfg)
            model_ref = Model(self._whole_cfg, use_value_network=False).state_dict()
            self.comm.model_ref = {k: val.cpu().share_memory_() for k, val in model_ref.items()}
            self.comm._register_learner(self, self._ip, self._port, self._rank, self.world_size)
            # self.comm.start_send_model()
            self._reset_value_flag = False
            self._update_config_flag = False
            self._reset_comm_setting_flag = False
            self._address_dir = os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                             self._player_id, 'address')
            self._config_dir = os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                            self.comm.player_id,'config')
            os.makedirs(self._address_dir, exist_ok=True)
            with open(os.path.join(self._address_dir, f'{self._ip}:{self._port}'), 'w') as f:
                f.write(f'rank:{self.rank}, ip:{self._ip}, port:{self._port},'
                        f' world_size:{self.world_size}'
                        f' player_id:{self._player_id}')
        self._remain_value_pretrain_iters = self._whole_cfg.learner.get('value_pretrain_iters', -1)
        self._train_iter = 0
        self._timer_all = EasyTimer(self._use_cuda)
        self._total_train_time = 0
        self._pre_train_finished = False
        self.tb_path = os.path.join(os.getcwd(), 'experiments', 'learner_speed', self._player_id)
        self._writer = SummaryWriter(self.tb_path)
        self._mc = MemCache()

    def _setup_model(self):
        self._model = Model(self._whole_cfg, use_value_network=True)

    def _setup_loss(self):
        self._loss = ReinforcementLoss(self._whole_cfg.learner,self._whole_cfg.learner.player_id)

    def _setup_optimizer(self):
        self._optimizer = Adam(
            self.model.parameters(),
            lr=self._whole_cfg.learner.learning_rate,
            betas=(0, 0.99),
            eps=1e-5,
        )
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[], gamma=1)

    def _train(self, data):
        self.step_value_pretrain()
        self._mc.show_cuda_info()
        self._mc['data'] = data

        with self._timer_all:
            with self._timer:
                self._mc['model_output'] = self._model.rl_learner_forward(**self._mc['data'])
                if self._whole_cfg.learner.use_dapo:
                    self._mc['model_output']['successive_logit'] = self._mc['data']['successive_logit']
                self._mc['log_vars'] = self._loss.compute_loss(self._mc['model_output'])

                self._mc['loss'] = self._mc['log_vars']['total_loss']
            self._log_buffer['forward_time'] = self._timer.value

            with self._timer:
                self._optimizer.zero_grad()
                self._mc['loss'].backward()
                if self._use_distributed:
                    self._model.sync_gradients()
                if self._save_grad and self._last_iter.val % self.save_log_freq == 0:
                    for k, param in self._model.named_parameters():
                        if param.grad is not None:
                            self.grad_tb_logger.add_scalar(k, (torch.norm(param.grad)).item(),
                                                        global_step=self._last_iter.val)
                            self.model_tb_logger.add_scalar(k, (torch.norm(param.data)).item(),
                                                            global_step=self._last_iter.val)
                self._mc['gradient'] = self._grad_clip.apply(self._model.parameters())
                if self._save_grad and self._last_iter.val % self.save_log_freq == 0:
                    for k, param in self._model.named_parameters():
                        if param.grad is not None:
                            self.clip_grad_tb_logger.add_scalar(k, (torch.norm(param.grad)).item(),
                                                        global_step=self._last_iter.val)

                self._optimizer.step()
            self._log_buffer['backward_time'] = self._timer.value
            # self._lr_scheduler.step()
        current_train_time = self._timer_all.value
        self._train_iter +=1 
        if self._train_iter >= PRE_TRAIN_ITER and self._pre_train_finished is False:
            self._pre_train_finished = True
            self._train_iter = 0
        
        if self._pre_train_finished:
            self._total_train_time += current_train_time
            logging.info(
                "[Learner] trained {} train_iter in total, current training speed is {} iter/s, total recv speed is {} iter/s".format(
                    self._train_iter,
                    1 / current_train_time,
                    self._train_iter / self._total_train_time
            ))
            self._writer.add_scalar("current_train_speed/iter_s", 1 / current_train_time, self._train_iter)
            self._writer.add_scalar("total_train_speed/iter_s", self._train_iter / self._total_train_time, self._train_iter)
        else:
            print('we are now pretraining', self._train_iter)
        self._log_buffer['gradient'] = self._mc['gradient']
        self._log_buffer.update(self._mc['log_vars'])
        if self._update_config_flag:
            self.update_config()
            self._update_config_flag = False
        if self._reset_value_flag:
            self.reset_value()
            self._reset_value_flag = False
        if self._reset_comm_setting_flag:
            self.reset_comm_setting()
            self._reset_comm_setting_flag = False
        self._mc.mclean()
        self._mc.show_cuda_info()

    def step_value_pretrain(self):
        if self._remain_value_pretrain_iters > 0:
            self._loss.only_update_value = True
            self._remain_value_pretrain_iters -= 1
            if self._use_distributed:
                self._model.module.only_update_baseline = True
            else:
                self._model.only_update_baseline = True

        elif self._remain_value_pretrain_iters == 0:
            self._loss.only_update_value = False
            self._remain_value_pretrain_iters -= 1
            if self._rank == 0:
                self._logger.info('value pretrain iter is 0')
            if self._use_distributed:
                self._model.module.only_update_baseline = False
            else:
                self._model.only_update_baseline = False

    def register_stats(self) -> None:
        """
        Overview:
            register some basic attributes to record & tb_logger(e.g.: cur_lr, data_time, train_time),
            register the attributes related to computation_graph to record & tb_logger.
        """
        super(RLLearner, self).register_stats()
        for k in ['total_loss', 'kl/extra_at', 'gradient']:
            self._record.register_var(k)
            self._tb_logger.register_var(k)

        for k1 in ['winloss', 'build_order', 'built_unit', 'effect', 'upgrade', 'battle', 'upgo', 'kl', 'entropy']:
            for k2 in ['reward', 'value', 'td', 'action_type', 'delay', 'queued', 'selected_units',
                       'target_unit', 'target_location', 'total']:
                k = k1 + '/' + k2
                self._record.register_var(k)
                self._tb_logger.register_var(k)

    def _setup_dataloader(self):
        if self._job_type == 'train':
            pass
        else:
            self._dataloader = FakeDataloader(unroll_len=self._whole_cfg.actor.traj_len,
                                              batch_size=self._whole_cfg.learner.data.batch_size)

    @property
    def cfg(self):
        return self._whole_cfg.learner

    def update_config(self):
        load_config_path = os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                        f'user_config.yaml')
        load_config = read_config(load_config_path)
        player_id = self._whole_cfg.learner.player_id
        self._whole_cfg = deep_merge_dicts(self._whole_cfg, load_config)
        self._whole_cfg.learner.player_id = player_id
        self._setup_loss()
        self._remain_value_pretrain_iters = self._whole_cfg.learner.get('value_pretrain_iters', -1)
        if self.use_distributed:
            self.model.module.lstm_traj_infer = self._whole_cfg.learner.get('lstm_traj_infer', False)
        else:
            self.model.lstm_traj_infer = self._whole_cfg.learner.get('lstm_traj_infer', False)
        for g in self._optimizer.param_groups:
            g['lr'] = self._whole_cfg.learner.learning_rate
        print(f'update config from config_path:{load_config_path}')
        if self.rank == 0:
            time_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
            config_path = os.path.join(self._config_dir, f'user_config_{time_label}.yaml')
            save_config(self._whole_cfg, config_path)
            print(f'save_config to config_path:{config_path}')

    def _reset_value(self):
        ref_model = Model(self._whole_cfg, use_value_network=True)
        value_state_dict = {k: val for k, val in ref_model.state_dict().items() if 'value' in k or 'auxiliary' in k}
        self.model.load_state_dict(value_state_dict, strict=False)

    def reset_value(self):
        flag = torch.tensor([0])
        if self.rank == 0:
            flag = torch.tensor([1])
            self._reset_value()
        if self.world_size > 1:
            broadcast(flag, 0)
            if flag:
                self._setup_optimizer()
                self.model.broadcast_params()
        elif self.world_size == 1:
            self._setup_optimizer()
        print(f'reset_value')

    def reset_comm_setting(self):
        self.comm.close()
        del self.comm
        self._dataloader.close()
        del self._dataloader
        self._reset_comm()
        self._reset_dataloader()

    def _reset_comm(self):
        self.comm = LearnerComm(self._whole_cfg)
        self.comm._register_learner(self, self._ip, self._port, self._rank, self.world_size)
        model_ref = Model(self._whole_cfg, use_value_network=False).state_dict()
        self.comm.model_ref = {k: val.cpu().share_memory_() for k, val in model_ref.items()}
        # self.comm.start_send_model()

    def _reset_dataloader(self):

        self._dataloader = RLDataLoader(data_source=self.comm.ask_for_metadata, cfg=self._whole_cfg)

    @staticmethod
    def create_rl_learner_app(learner):
        app = Flask(__name__)

        def build_ret(code, info=''):
            return {'code': code, 'info': info}

        # ************************** debug use *********************************
        @app.route('/rl_learner/update_config', methods=['GET'])
        def learner_update_config():
            learner._update_config_flag = True
            return {"done": "successfuly update config"}

        @app.route('/rl_learner/reset_comm_setting', methods=['GET'])
        def learner_reset_comm_setting():
            learner._update_config_flag = True
            learner._reset_comm_setting_flag = True
            return {"done": "successfuly reset_comm_setting"}

        @app.route('/rl_learner/reset_value', methods=['GET'])
        def learner_reset_value():
            learner._reset_value_flag = True
            learner._update_config_flag = True
            return {"done": "successfuly reset_value and update config"}
        return app


class SendModelHook(LearnerHook):
    def __init__(self, name='send_model_hook', position='after_iter', priority=40):
        super(SendModelHook, self).__init__(name=name, position=position, priority=priority)

    def __call__(self, engine):
        pass
        # if self.position == 'before_run':
        #     engine.comm.send_model(engine, ignore_freq=True)
        # elif self.position == 'after_iter':
        #     engine.comm.send_model(engine)


class SendTrainInfo(LearnerHook):
    def __init__(self, name='send_train_info_hook', position='after_iter', priority=60):
        super(SendTrainInfo, self).__init__(name=name, position=position, priority=priority)

    def __call__(self, engine):
        engine.comm.send_train_info(engine)

