
import os
import time

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# PyTorch imports
import torch
from torch.optim import Adam
import urllib.request
import pandas as pd


# Create 'utils' directory and download required files if it doesn't exist
utils_dir = "utils"
if not os.path.isdir(utils_dir):
    os.makedirs(utils_dir)
    print(f"Directory '{utils_dir}' created.")

    files = {"logx.py": "https://raw.githubusercontent.com/alibaniasad1999/spinningup/master/spinup/utils/logx.py",
        "mpi_tools.py": "https://raw.githubusercontent.com/alibaniasad1999/spinningup/master/spinup/utils/mpi_tools.py",
        "serialization_utils.py": "https://raw.githubusercontent.com/alibaniasad1999/spinningup/master/spinup/utils/serialization_utils.py",
        "run_utils.py": "https://raw.githubusercontent.com/alibaniasad1999/spinningup/master/spinup/utils/run_utils.py",
        "user_config.py": "https://raw.githubusercontent.com/alibaniasad1999/spinningup/master/spinup/user_config.py",
         "mpi_pytorch.py": "https://raw.githubusercontent.com/alibaniasad1999/spinningup/master/spinup/utils/mpi_pytorch.py"}

    for filename, url in files.items():
        dest = os.path.join(utils_dir, filename)
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"{filename} downloaded.")
else:
    print(f"Directory '{utils_dir}' already exists.")

import numpy as np
import scipy.signal
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from utils.logx import EpochLogger

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, deterministic=False):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        if deterministic:
            epsilon = 1e-6
            std = torch.zeros_like(std) + epsilon
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, deterministic=False):
        with torch.no_grad():
            pi = self.pi._distribution(obs, deterministic=deterministic)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]




class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.act_buf_1 = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf_1 = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf_1 = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf_1 = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.val_buf_1 = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf_1 = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, act_1, rew_1, val_1, logp_1):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.act_buf_1[self.ptr] = act_1
        self.rew_buf[self.ptr] = rew
        self.rew_buf_1[self.ptr] = rew_1
        self.val_buf[self.ptr] = val
        self.val_buf_1[self.ptr] = val_1
        self.logp_buf[self.ptr] = logp
        self.logp_buf_1[self.ptr] = logp_1
        self.ptr += 1

    def finish_path(self, last_val=0, last_val_1=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be zero if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        rews_1 = np.append(self.rew_buf_1[path_slice], last_val_1)
        vals = np.append(self.val_buf[path_slice], last_val)
        vals_1 = np.append(self.val_buf_1[path_slice], last_val_1)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        deltas_1 = rews_1[:-1] + self.gamma * vals_1[1:] - vals_1[:-1]
        self.adv_buf_1[path_slice] = discount_cumsum(deltas_1, self.gamma)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.ret_buf_1[path_slice] = discount_cumsum(rews_1, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, player = 0):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        if player == 0:
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
            self.adv_buf = (self.adv_buf - adv_mean) / adv_std
            data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                        adv=self.adv_buf, logp=self.logp_buf)
        else:
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf_1)
            self.adv_buf_1 = (self.adv_buf_1 - adv_mean) / adv_std
            data = dict(obs=self.obs_buf, act_1=self.act_buf, ret_1=self.ret_buf,
                        adv_1=self.adv_buf_1, logp_1=self.logp_buf_1)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}



class PPO:
    def __init__(self, env, ac_kwargs=None, seed=0,
        steps_per_epoch=30000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=30000,
        target_kl=0.01, logger_kwargs=None, save_freq=10):
        self.env = env
        self.ac_kwargs = ac_kwargs or {}
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.logger_kwargs = logger_kwargs or {}
        setup_pytorch_for_mpi()
        self.save_freq = save_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """
            Proximal Policy Optimization (by clipping),

            with early stopping based on approximate KL

            Args:
                env : The environment must satisfy the OpenAI Gym API.

                actor_critic: The constructor method for a PyTorch Module with a
                    ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
                    module. The ``step`` method should accept a batch of observations
                    and return:

                    ===========  ================  ======================================
                    Symbol       Shape             Description
                    ===========  ================  ======================================
                    ``a``        (batch, act_dim)  | Numpy array of actions for each
                                                   | observation.
                    ``v``        (batch,)          | Numpy array of value estimates
                                                   | for the provided observations.
                    ``logp_a``   (batch,)          | Numpy array of log probs for the
                                                   | actions in ``a``.
                    ===========  ================  ======================================

                    The ``act`` method behaves the same as ``step`` but only returns ``a``.

                    The ``pi`` module's forward call should accept a batch of
                    observations and optionally a batch of actions, and return:

                    ===========  ================  ======================================
                    Symbol       Shape             Description
                    ===========  ================  ======================================
                    ``pi``       N/A               | Torch Distribution object, containing
                                                   | a batch of distributions describing
                                                   | the policy for the provided observations.
                    ``logp_a``   (batch,) | Optional (only returned if batch of
                                                   | actions is given). Tensor containing
                                                   | the log probability, according to
                                                   | the policy, of the provided actions.
                                                   | If actions not given, will contain
                                                   | ``None``.
                    ===========  ================  ======================================

                    The ``v`` module's forward call should accept a batch of observations
                    and return:

                    ===========  ================  ======================================
                    Symbol       Shape             Description
                    ===========  ================  ======================================
                    ``v``        (batch,)          | Tensor containing the value estimates
                                                   | for the provided observations. (Critical:
                                                   | make sure to flatten this!)
                    ===========  ================  ======================================


                ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                    you provided to PPO.

                seed (int): Seed for random number generators.

                steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                    for the agent and the environment in each epoch.

                epochs (int): Number of epochs of interaction (equivalent to
                    number of policy updates) to perform.

                gamma (float): Discount factor. (Always between 0 and 1.)

                clip_ratio (float): Hyperparameter for clipping in the policy objective.
                    Roughly: how far can the new policy go from the old policy while
                    still profiting (improving the objective function)? The new policy
                    can still go farther than the clip_ratio says, but it doesn't help
                    on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
                    denoted by :math:`\epsilon`.

                pi_lr (float): Learning rate for policy optimizer.

                vf_lr (float): Learning rate for value function optimizer.

                train_pi_iters (int): Maximum number of gradient descent steps to take
                    on policy loss per epoch. (Early stopping may cause optimizer
                    to take fewer than this.)

                train_v_iters (int): Number of gradient descent steps to take on
                    value function per epoch.

                lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                    close to 1.)

                max_ep_len (int): Maximum length of trajectory / episode / rollout.

                target_kl (float): Roughly what KL divergence we think is appropriate
                    between new and old policies after an update. This will get used
                    for early stopping. (Usually small, 0.01 or 0.05.)

                logger_kwargs (dict): Keyword args for EpochLogger.

                save_freq (int): How often (in terms of gap between epochs) to save
                    the current policy and value function.

            """
        # Set up logger and save configuration
        self.logger = EpochLogger(**self.logger_kwargs)
        self.logger.save_config(locals())

        # Random seed
        self.seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate environment
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape

        # Create actor-critic module
        self.ac = MLPActorCritic(self.env.observation_space, self.env.action_space, **self.ac_kwargs)
        # second player
        self.ac_1 = MLPActorCritic(self.env.observation_space, self.env.action_space, **self.ac_kwargs)

        # Sync params across processes
        sync_params(self.ac)

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.logger.log('\nNumber of parameters 😱😱😱: \t pi: %d, \t v: %d\n' % var_counts)

        # Set up experience buffer
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.buf = PPOBuffer(obs_dim, act_dim, self.local_steps_per_epoch, gamma, lam)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        # second player
        self.pi_optimizer_1 = Adam(self.ac_1.pi.parameters(), lr=pi_lr)
        self.vf_optimizer_1 = Adam(self.ac_1.pi.parameters(), lr=vf_lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)
        self.logger.setup_pytorch_saver(self.ac_1)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()


    def update(self, first_player=True)
        data = self.buf.get()
        data_1 = self.buf.get(player=1)

        pi_l_old, pi_info_old = self.compute_loss_pi(data) # Loss pi before
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()
        # second player
        pi_l_old_1, pi_info_old_1 = self.compute_loss_pi(data_1)
        pi_l_old_1 = pi_l_old_1.item()
        v_l_old_1 = self.compute_loss_v(data_1).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            if not first_player:
                break
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
            self.pi_optimizer.step()
        # second player learning loop
        for i in range(self.train_pi_iters):
            self.pi_optimizer_1.zero_grad()
            loss_pi, pi_info_1 = self.compute_loss_pi(data_1)
            kl = mpi_avg(pi_info_1['kl'])
            if kl > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(self.ac_1.pi)
            self.pi_optimizer_1.step()

        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            if not first_player:
                break
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)  # average grads across MPI processes
            self.vf_optimizer.step()
        # second player learning loop
        for i in range(self.train_v_iters):
            self.vf_optimizer_1.zero_grad()
            loss_v_1 = self.compute_loss_v(data)
            loss_v_1.backward()
            mpi_avg_grads(self.ac.v)
            self.vf_optimizer_1.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        kl_1, ent_1, cf_1 = pi_info_1['kl'], pi_info_old_1['ent'], pi_info_1['cf']
        if not first_player:
            self.logger.store(LossPi=0, LossV=0,
                            KL=0, Entropy=0, ClipFrac=0,
                            DeltaLossPi=0,
                            DeltaLossV=0,
                            LossPi_1=pi_l_old_1, LossV_1=v_l_old_1,
                            KL_1=kl_1, Entropy_1=ent_1, ClipFrac_1=cf_1,
                            DeltaLossPi_1=(loss_pi_1.item() - pi_l_old_1),
                            DeltaLossV_1=(loss_v_1.item() - v_l_old_1))
        else:
            self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                            KL=kl, Entropy=ent, ClipFrac=cf,
                            DeltaLossPi=(loss_pi.item() - pi_l_old),
                            DeltaLossV=(loss_v.item() - v_l_old),
                            LossPi_1=pi_l_old_1, LossV_1=v_l_old_1,
                            KL_1=kl_1, Entropy_1=ent_1, ClipFrac_1=cf_1,
                            DeltaLossPi_1=(loss_pi_1.item() - pi_l_old_1),
                            DeltaLossV_1=(loss_v_1.item() - v_l_old_1))

    def train(self, first_player_learning_epoch=50):
        # Prepare for interaction with environment
        start_time = time.time()
        o, _ = self.env.reset()
        ep_ret, ep_len = 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            for t in range(self.local_steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                a_1, v_1, logp_1 = self.ac_1.step(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, _, _ = self.env.step(a, a_1)
                ep_ret += r
                ep_len += 1

                # save and log act_1, rew_1, val_1, logp_1
                self.buf.store(o, a, r, v, logp, a_1, -r, v_1, logp_1)
                self.logger.store(VVals=v)
                self.logger.store(VVals_1=v_1)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.local_steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                        _, v_1, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                        v_1 = 0
                    self.buf.finish_path(v, v_1)
                    if terminal:
                        # only save EpRet / EpLen if the trajectory finished
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, _ = self.env.reset()
                    ep_ret, ep_len = 0, 0

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                self.logger.save_state({'env': self.env}, None)

            # Perform PPO update!
            self.update()

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.steps_per_epoch)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('ClipFrac', average_only=True)
            self.logger.log_tabular('StopIter', average_only=True)
            self.logger.log_tabular('Time', time.time() - start_time)
            self.logger.dump_tabular()


    def test(self, fun_mode=False, deterministic=True, save_data=False):
        o, _ = self.env.reset()
        states = []
        actions = []
        while True:
            a, _, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32), deterministic=deterministic)
            actions.append(a)
            o, _, d, _, position = self.env.step(a)
            states.append(position)
            if d:
                break
        dt = self.env.dt
        time = np.arange(0, len(states)*dt, dt)
        state_array = np.array(states)
        action_array = np.array(actions)

        # save trajectory and actions to csv
        if not os.path.exists('results/') and save_data:
            os.makedirs('results/')

        # numpy to pandas with header
        state_df = pd.DataFrame(state_array, columns=['x', 'y', 'xdot', 'ydot'])
        action_df = pd.DataFrame(action_array, columns=['ax', 'ay'])

        # save to csv
        if save_data:
            state_df.to_csv('results/state.csv', index=False)
            action_df.to_csv('results/action.csv', index=False)
            print(colorize("Data saved to results folder 😜", 'green', bold=True))

        df = pd.read_csv('trajectory.csv')
        # df to numpy array
        data = df.to_numpy()
        print(data.shape)
        trajectory = np.delete(data, 2, 1)
        trajectory = np.delete(trajectory, -1, 1)

        if fun_mode:
            # Use XKCD style for hand-drawn look
            with plt.xkcd():
                plt.plot(state_array[:, 0], state_array[:, 1], label='State')
                plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
                plt.legend()
                plt.show()
            with plt.xkcd():
                plt.plot(time, action_array)
                plt.xlabel("Time (sec)")
                plt.ylabel("action (N)")
                plt.show()
        else:
            plt.plot(state_array[:, 0], state_array[:, 1], label='State')
            plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
            plt.legend()
            # axis equalor
            plt.axis('equal')

            plt.show()

            plt.plot(action_array)
            plt.xlabel("Time (sec)")
            plt.ylabel("action (N)")
            plt.show()# save trajectory and actions to csv
        if not os.path.exists('results/') and save_data:
            os.makedirs('results/')



    # save actor critic
    def save(self, filepath='model/'):
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        # Check the device_ of the model
        if self.device == 'cuda':
            torch.save(self.ac.pi.state_dict(), filepath + 'actor_cuda.pth')
            torch.save(self.ac.v.state_dict(), filepath + 'v_cuda.pth')
        else:
            torch.save(self.ac.pi.state_dict(), filepath + 'actor_cpu.pth')
            torch.save(self.ac.v.state_dict(), filepath + 'v_cpu.pth')
        print(colorize(f"Model saved successfully! 🥰😎", 'blue', bold=True))

    # load actor critic
    def load(self, filepath='model/', load_device=torch.device("cpu"), from_device_to_load='cpu'):
        self.start_steps = 0  # does not distarct the loaded model
        # check if the model is available
        if os.path.isfile(filepath + 'actor_cpu.pth') or os.path.isfile(filepath + 'actor_cuda.pth'):
            # Check the device_ of the model
            if from_device_to_load == 'cpu':
                actor_file = 'actor_cpu.pth'
                v_file = 'v_cpu.pth'
            else:
                actor_file = 'actor_cuda.pth'
                v_file = 'v_cuda.pth'

            if from_device_to_load == 'cpu' and load_device.type == 'cuda':
                self.ac.pi.load_state_dict(torch.load(filepath + actor_file, map_location=torch.device('cuda')))
                self.ac.v.load_state_dict(torch.load(filepath + v_file, map_location=torch.device('cuda')))
            elif from_device_to_load == 'cuda' and load_device.type == 'cpu':
                self.ac.pi.load_state_dict(torch.load(filepath + actor_file, map_location=torch.device('cpu')))
                self.ac.v.load_state_dict(torch.load(filepath + v_file, map_location=torch.device('cpu')))
            else:
                self.ac.pi.load_state_dict(torch.load(filepath + actor_file))
                self.ac.v.load_state_dict(torch.load(filepath + v_file))
            print(colorize(f"Model loaded successfully and device is {load_device}! 🥰😎", 'blue', bold=True))
        else:
            print(colorize("Model not found! 😱🥲", 'red', bold=True))
