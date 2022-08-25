import os
from typing import Callable, Tuple, List, Any

import numpy as np
import torch
import torch.distributed as dist

from .default_helper import error_wrapper

# from .slurm_helper import get_master_addr


def get_rank() -> int:
    r"""
    Overview:
        Get the rank of current process in total world_size
    """
    # return int(os.environ.get('SLURM_PROCID', 0))
    return error_wrapper(dist.get_rank, 0)()


def get_world_size() -> int:
    r"""
    Overview:
        Get the world_size(total process number in data parallel training)
    """
    # return int(os.environ.get('SLURM_NTASKS', 1))
    return error_wrapper(dist.get_world_size, 1)()


broadcast = dist.broadcast
allgather = dist.all_gather


def allreduce(x: torch.Tensor) -> None:
    dist.all_reduce(x)
    x.div_(get_world_size())


def allreduce_async(name: str, x: torch.Tensor) -> None:
    x.div_(get_world_size())
    dist.all_reduce(x, async_op=True)


synchronize = torch.cuda.synchronize


def get_group(group_size: int) -> List:
    r"""
    Overview:
        Get the group segmentation of ``group_size`` each group
    Arguments:
        - group_size (:obj:`int`) the ``group_size``
    """
    rank = get_rank()
    world_size = get_world_size()
    if group_size is None:
        group_size = world_size
    assert (world_size % group_size == 0)
    return simple_group_split(world_size, rank, world_size // group_size)


def dist_mode(func: Callable) -> Callable:
    r"""
    Overview:
        Wrap the function so that in can init and finalize automatically before each call
    """

    def wrapper(*args, **kwargs):
        dist_init()
        func(*args, **kwargs)
        dist_finalize()

    return wrapper


def dist_init(backend: str = 'nccl',
              addr: str = None,
              port: str = None,
              rank: int = None,
              world_size: int = None) -> Tuple[int, int]:
    r"""
    Overview:
        Init the distributed training setting
    """
    assert backend in ['nccl', 'gloo'], backend
    os.environ['MASTER_ADDR'] = addr or os.environ.get('MASTER_ADDR', "localhost")
    os.environ['MASTER_PORT'] = port or os.environ.get('MASTER_PORT', "10314")  # hard-code

    if rank is None:
        local_id = os.environ.get('SLURM_PROCID', 0)
        if local_id is None:
            raise RuntimeError("please indicate rank explicitly in dist_init method")
        else:
            rank = int(local_id)
    if world_size is None:
        ntasks = os.environ.get('SLURM_NTASKS', os.environ.get('WORLD_SIZE', None))
        if ntasks is None:
            raise RuntimeError("please indicate world_size explicitly in dist_init method")
        else:
            world_size = int(ntasks)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    world_size = get_world_size()
    rank = get_rank()
    return rank, world_size


def dist_finalize() -> None:
    r"""
    Overview:
        Finalize distributed training resources
    """
    dist.destroy_process_group()


class DistContext:

    def __init__(self) -> None:
        pass

    def __enter__(self) -> None:
        dist_init()

    def __exit__(self, *args, **kwargs) -> Any:
        dist_finalize()


def simple_group_split(world_size: int, rank: int, num_groups: int) -> List:
    r"""
    Overview:
        Split the group according to ``worldsize``, ``rank`` and ``num_groups``

    .. note::
        With faulty input, raise ``array split does not result in an equal division``
    """
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(dist.new_group(rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank // group_size]


class DistModule(torch.nn.Module):
    r"""
    Overview:
        Distributed module that wrapped the nn.model
    Interface:
        __init__, sync_gradients, broadcast_params
    """

    def __init__(self, module, sync=True):
        r"""
        Overview:
            init method of the DistModule
        Arguments:
            - module (:obj:`nn.model`): the module to be wrapped
            - sync (:obj:`bool`): whether need syncronize
        """
        super(DistModule, self).__init__()
        self.module = module
        self._extend_module_attr()
        self.broadcast_params()

        self.sync = sync
        if not sync:
            self._grad_accs = []
            self._register_hooks()

    def _extend_module_attr(self):
        # if you want to use more attributes of torch.nn.module, please extend this module
        # and overwrite this method or let the repo developer informed.
        attributes = ['forward', 'state_dict', 'load_state_dict', 'named_parameters', 'sl_train','rl_learner_forward']
        for attr in attributes:
            setattr(self, attr, getattr(self.module, attr))

    def _create_grad(self):
        for name, param in self.named_parameters():
            setattr(param, 'grad', torch.zeros_like(param))

    def _register_hooks(self):
        for i, (name, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)

    def _make_hook(self, name, p, i):

        def hook(*ignore):
            dist.all_reduce(name, p.grad.data)

        return hook

    def sync_gradients(self):
        r"""
        Overview:
            calculate the average gradients
        """
        if self.sync and dist.get_world_size() > 1:
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    allreduce(param.grad.data)
        else:
            torch.cuda.synchronize()

    def broadcast_params(self):
        """
        Overview:
            broadcast the model parameters
        """
        for name, param in self.state_dict().items():
            dist.broadcast(param, 0)
