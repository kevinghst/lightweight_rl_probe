from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import time

import pdb

from contextlib import nullcontext
from tqdm import tqdm
import copy
import wandb
import numpy as np
from src.scrl import *
from src.utils import select_at_indexes, find_weight_norm, DataWriter
from rlpyt.utils.collections import namedarraytuple
from collections import namedtuple
from src.utils import discount_return_n_step
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from src.rlpyt_buffer import AsyncPrioritizedSequenceReplayFrameBufferExtended, \
    AsyncUniformSequenceReplayFrameBufferExtended
from rlpyt.replays.sequence.prioritized import SamplesFromReplayPri

from src.losses import SigmoidFocalLoss, SoftmaxFocalLoss

from sklearn.metrics import precision_score, recall_score, f1_score
from cyanure.estimators import Classifier
from cyanure.data_processing import preprocess

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])
ModelSamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                       ["observation", "action", "reward", "done", "value"])

OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr"])
ModelOptInfo = namedtuple("OptInfo", ["loss", "gradNorm",
                                      "tdAbsErr",
                                      "GoalLoss",
                                      "GoalError",
                                      "modelGradNorm",
                                      "T0SPRLoss",
                                      "InverseModelLoss",
                                      "RewardLoss",
                                      "BCLoss",
                                      "SampleTime",
                                      "ForwardTime",
                                      "CNNWeightNorm",
                                      "ModelSPRLoss",
                                      "Diversity",
                                      "KLLoss",
                                      "EmbedCos",
                                      "ProjCos"
                                      ])

EPS = 1e-6  # (NaN-guard)


class SPRCategoricalDQN(CategoricalDQN):
    """Distributional DQN with fixed probability bins for the Q-value of each
    action, a.k.a. categorical."""

    def __init__(self,
                 rl_weight=1.,
                 spr_weight=1.,
                 inverse_model_weight=1,
                 t0_weight=0,
                 goal_n_step=1,
                 goal_window=50,
                 goal_weight=1.,
                 goal_permute_prob=0.2,
                 goal_noise_weight=0.5,
                 goal_reward_scale=5.,
                 goal_all_to_all=False,
                 conv_goal=False,
                 clip_model_grad_norm=10.,
                 goal_dist="exp",
                 jumps=0,
                 offline=False,
                 bc_weight=0,
                 kl_weight=0.1,
                 probe='prior',
                 probe_jumps=[],
                 probe_control=False,
                 probe_condition=1,
                 probe_task='reward',
                 supervised=False,
                 lr_ft=0.1,
                 weight_decay_ft=0.000001,
                 epoch_ft=1,
                 lr_drop_ft=10,
                 clip_max_norm_ft=0.1,
                 reward_ft_weight=1,
                 term_ft_weight=1,
                 encoder_lr: Optional[float] = None,
                 dynamics_model_lr: Optional[float] = None,
                 q_l1_lr: Optional[float] = None,
                 data_writer_args={"save_data": False},
                 eval_only=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.opt_info_fields = tuple(f for f in ModelOptInfo._fields)  # copy
        self.spr_weight = spr_weight
        self.inverse_model_weight = inverse_model_weight
        self.t0_weight = t0_weight
        self.clip_model_grad_norm = clip_model_grad_norm
        self.goal_window = goal_window
        self.goal_n_step = goal_n_step
        self.goal_permute_prob = goal_permute_prob
        self.goal_reward_scale = goal_reward_scale
        self.goal_noise_weight = goal_noise_weight
        self.goal_all_to_all = goal_all_to_all
        self.offline = offline
        self.conv_goal = conv_goal

        self.probe = '' if probe == 'none' else probe
        self.probe_jumps = probe_jumps
        self.probe_condition = probe_condition
        self.probe_control = probe_control
        self.probe_task = probe_task
        self.eval_only = eval_only

        self.supervised = supervised
        self.lr_ft = lr_ft
        self.weight_decay_ft = weight_decay_ft
        self.epoch_ft = epoch_ft
        self.lr_drop_ft = lr_drop_ft
        self.clip_max_norm_ft = clip_max_norm_ft
        self.reward_ft_weight = reward_ft_weight
        self.term_ft_weight = term_ft_weight
        self.kl_weight = kl_weight

        self.bc_weight = bc_weight

        if "exp" in goal_dist.lower():
            self.goal_distance = exp_distance
        else:
            self.goal_distance = norm_dist

        self.rl_weight = rl_weight
        self.goal_weight = goal_weight
        self.jumps = jumps

        self.save_data = data_writer_args["save_data"]
        if self.save_data:
            self.data_writer = DataWriter(**data_writer_args)

        self.encoder_lr = encoder_lr if encoder_lr is not None else self.learning_rate
        self.dynamics_model_lr = dynamics_model_lr if dynamics_model_lr is not None else self.learning_rate
        self.q_l1_lr = q_l1_lr if q_l1_lr is not None else self.learning_rate

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        example_to_buffer = ModelSamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            value=examples["agent_info"].p,
        )
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            batch_T=max(self.jumps + 1, self.goal_window) if self.goal_weight > 0 else self.jumps + 1,
            discount=self.discount,
            n_step_return=self.n_step_return,
            rnn_state_interval=0,
        )

        if self.prioritized_replay:
            replay_kwargs['alpha'] = self.pri_alpha
            replay_kwargs['beta'] = self.pri_beta_init
            # replay_kwargs["input_priorities"] = self.input_priorities
            buffer = AsyncPrioritizedSequenceReplayFrameBufferExtended(**replay_kwargs)
        else:
            buffer = AsyncUniformSequenceReplayFrameBufferExtended(**replay_kwargs)

        self.replay_buffer = buffer

    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank
        try:
            # We're probably dealing with DDP
            self.model = self.agent.model.module
        except:
            self.model = self.agent.model

        # Split into (optionally) three groups for separate LRs.
        conv_params, dynamics_model_params, q_l1_params, other_params = self.model.list_params()
        self.optimizer = self.OptimCls([
            {'params': conv_params, 'lr': self.encoder_lr},
            {'params': q_l1_params, 'lr': self.q_l1_lr},
            {'params': dynamics_model_params, 'lr': self.dynamics_model_lr},
            {'params': other_params, 'lr': self.learning_rate}
        ],
            **self.optim_kwargs)

        if self.initial_optim_state_dict is not None and not self.eval_only:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        if self.prioritized_replay:
            self.pri_beta_itr = max(1, self.pri_beta_steps // self.sampler_bs)

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method.  In
        asynchronous mode, will be called in the memory_copier process."""
        return ModelSamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
            value=samples.agent.agent_info.p,
        )

    def sample_batch(self):
        if not self.offline:
            samples = self.replay_buffer.sample_batch(self.batch_size)
            return samples
        else:
            return self.sample_offline_dataset()

    def sample_offline_dataset(self):
        try:
            samples = next(self.offline_dataloader)
        except Exception as e:
            self.offline_dataloader = iter(self.offline_dataset)
            samples = next(self.offline_dataloader)
        return samples

    def build_optimizer_ft(self):
        if self.supervised:
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad]}
            ]
        else:
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and "_ft" in n]}
            ]
        # TODO: optimize for lr rate
        optimizer = torch.optim.SGD(param_dicts, lr=self.lr_ft, momentum=0.9, weight_decay=self.weight_decay_ft)
        lr_scheduler = StepLR(optimizer, self.lr_drop_ft)
        return optimizer, lr_scheduler

    def probe_features(self):
        if self.probe_task == 'reward':
            return self.probe_reward()
        else:
            return self.probe_action()

    def eval_linear(self, x, y):
        print("evaluating...")

        tracker = {
            'embed': {
                'total_is': [],
                'total_predicted': [],
                'total_loss': []
            }
        }

        self.model.eval()
        focal_loss = SoftmaxFocalLoss(gamma=2)

        for i, features in enumerate(x):
            label = y[i]
            with torch.no_grad():

                next_action_logits = self.model.embed_reward_predictor_ft(features)
            loss = focal_loss.compute_loss(next_action_logits, label)
            predicted = torch.argmax(next_action_logits, dim=1) # int64
            tracker['embed']['total_predicted'].append(predicted)
            tracker['embed']['total_is'].append(label)
            tracker['embed']['total_loss'].append(loss)

        ft_metrics = {}

        for jump, attr in tracker.items():
            total_is = torch.cat(attr['total_is']).cpu()
            total_predicted = torch.cat(attr['total_predicted']).cpu()
            ft_metrics[f'next_action_f1_{jump}'] = f1_score(total_is, total_predicted, average='weighted')

        return ft_metrics
    
    def train_linear(self, x, y):
        self.model.train()
        print(f'finetuning')

        # reinitialize ft layer weights
        self.model.reward_predictor_ft = copy.deepcopy(self.model.init_reward_predictor_ft)
        self.model.embed_reward_predictor_ft = copy.deepcopy(self.model.init_reward_predictor_ft)
        focal_loss = SoftmaxFocalLoss(gamma=2)

        optimizer, lr_scheduler = self.build_optimizer_ft()
        for epoch in range(self.epoch_ft):
            for i, features in enumerate(x):
                label = y[i]
                next_action_logits = self.model.embed_reward_predictor_ft(features)
                loss = focal_loss.compute_loss(next_action_logits, label)
                optimizer.zero_grad()
                loss.backward()

                if self.clip_max_norm_ft > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm_ft)
                optimizer.step()

            lr_scheduler.step()

    def probe_action(self):
        print('generating train features...')
        X_train, y_train = self.gen_features_for_probing(self.offline_ft_dataset)
        print('generating val features...')
        X_val, y_val = self.gen_features_for_probing(self.offline_eval_dataset)

        self.train_linear(X_train, y_train)
        ft_metrics = self.eval_linear(X_val, y_val)
        print(ft_metrics)
        return ft_metrics

    def probe_reward(self):
        print('generating train features...')
        X_train, y_train = self.gen_features_for_probing(self.offline_ft_dataset)
        print('generating val features...')
        X_val, y_val = self.gen_features_for_probing(self.offline_eval_dataset)
        y_val = np.expand_dims(y_val, 1)

        # do logistic regression
        classifier = Classifier(
            loss="logistic",
            penalty="l2",
            max_iter=300,
            tol=1e-5,
            verbose=True,
            fit_intercept=False,
            lambda_1=.000000004,
        )

        preprocess(X_train,centering=True,normalize=True,columns=False)
        preprocess(X_val,centering=True,normalize=True,columns=False)

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_val)

        ft_metrics = {}
        f1 = f1_score(y_val, y_pred)
        ft_metrics[f'reward_f1_embed'] = f1

        print(ft_metrics)
        return ft_metrics

    def gen_features_for_probing(self, dataset):
        dataloader = iter(dataset)

        total_latents = []
        total_rewards = []
        total_actions = []

        for i, samples in tqdm(enumerate(dataloader)):
            observations = samples.all_observation.to(self.agent.device)  # [6, 256, 4, 1, 84, 84]
            actions = samples.all_action.to(self.agent.device)
            rewards = samples.all_reward.to(self.agent.device)

            observations = observations.transpose(0, 1).flatten(2, 3)
            # [7, 256, 4, 1, 84, 84] --> [256, 7, 4, 84, 84]

            observations = self.model.transform(observations[:,0], self.model.eval_transforms)
            with nullcontext() if self.supervised else torch.no_grad():
                latents = self.model.stem_forward(observations)[0].flatten(1)

            total_latents.append(latents)
            total_rewards.append(rewards[1])
            total_actions.append(actions[1])

        if self.probe_task == 'reward':
            # put data on CPU for logistic regression
            total_latents = torch.cat(total_latents).cpu().numpy()
            total_rewards = torch.cat(total_rewards).cpu().numpy() != 0
            return total_latents, total_rewards
        else:
            return total_latents, total_actions

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Extracts the needed fields from input samples and stores them in the
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).  If using prioritized
        replay, updates the priorities for sampled training batches.
        """
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.=
        if samples is not None:
            if self.save_data:
                self.data_writer.write(samples)
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = ModelOptInfo(*([] for _ in range(len(ModelOptInfo._fields))))
        if not self.offline and itr < self.min_itr_learn:
            return opt_info
        for _ in range(1 if self.offline else self.updates_per_optimize):
            start = time.time()
            samples_from_replay = self.sample_batch()
            end = time.time()
            sample_time = end - start

            forward_time = time.time()
            rl_loss, td_abs_errors, goal_loss, \
            t0_spr_loss, model_spr_loss, \
            diversity, inverse_model_loss, bc_loss, \
            goal_abs_errors, kl_loss, embed_cos, proj_cos = self.loss(samples_from_replay, self.offline)
            forward_time = time.time() - forward_time
            total_loss = self.rl_weight * rl_loss
            total_loss += self.t0_weight * t0_spr_loss
            total_loss += self.spr_weight * model_spr_loss
            total_loss += self.goal_weight * goal_loss
            total_loss += self.inverse_model_weight * inverse_model_loss
            total_loss += self.bc_weight * bc_loss
            total_loss += self.kl_weight * kl_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            stem_params, model_params = self.model.split_stem_model_params()
            if self.clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(stem_params,
                                                           self.clip_grad_norm)
            else:
                grad_norm = 0
            if self.clip_model_grad_norm > 0:
                model_grad_norm = torch.nn.utils.clip_grad_norm_(model_params,
                                                                 self.clip_model_grad_norm)
            else:
                model_grad_norm = 0

            cnn_weight_norm = find_weight_norm(self.model.conv.parameters())

            self.optimizer.step()

            if not self.offline and self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(td_abs_errors)
            opt_info.loss.append(rl_loss.item())
            opt_info.gradNorm.append(
                torch.tensor(grad_norm).item())
            opt_info.GoalLoss.append(goal_loss.item())
            opt_info.modelGradNorm.append(torch.tensor(model_grad_norm).item())
            opt_info.T0SPRLoss.append(t0_spr_loss.item())
            opt_info.InverseModelLoss.append(inverse_model_loss.item())
            opt_info.BCLoss.append(bc_loss.item())
            opt_info.CNNWeightNorm.append(cnn_weight_norm.item())
            opt_info.SampleTime.append(sample_time)
            opt_info.ForwardTime.append(forward_time)
            opt_info.Diversity.append(diversity.item())
            opt_info.ModelSPRLoss.append(model_spr_loss.item())
            opt_info.tdAbsErr.extend(td_abs_errors[::8].cpu().numpy())
            opt_info.GoalError.extend(goal_abs_errors[::8].cpu().numpy())
            opt_info.KLLoss.append(kl_loss.item())
            opt_info.EmbedCos.append(embed_cos.item())
            opt_info.ProjCos.append(proj_cos.item())
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)
        self.update_itr_hyperparams(itr)
        return opt_info

    def rl_loss(self, log_pred_ps, observations, goals, actions, rewards, nonterminals, returns, index, n_step):
        """
        Computes the Distributional Q-learning loss, based on projecting the
        discounted rewards + target Q-distribution into the current Q-domain,
        with cross-entropy loss.

        Returns loss and KL-divergence-errors for use in prioritization.
        """
        delta_z = (self.V_max - self.V_min) / (self.agent.n_atoms - 1)
        z = torch.linspace(self.V_min, self.V_max, self.agent.n_atoms, device=log_pred_ps.device)
        next_z = z * (self.discount ** n_step)
        next_z = torch.ger(nonterminals[index], next_z)
        ret = returns[index].unsqueeze(-1)

        next_z = torch.clamp(ret + next_z, self.V_min, self.V_max)

        z_bc = z.view(1, -1, 1)
        next_z_bc = next_z.unsqueeze(-2)
        abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
        projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)

        with torch.no_grad():
            target_ps = self.agent.target(observations[index + n_step],
                                          actions[index + n_step],
                                          rewards[index + n_step],
                                          goals)
            if self.double_dqn:
                next_ps = self.agent(observations[index + n_step],
                                     actions[index + n_step],
                                     rewards[index + n_step],
                                     goals)
                next_qs = torch.tensordot(next_ps, z, dims=1)
                next_a = torch.argmax(next_qs, dim=-1)
            else:
                target_qs = torch.tensordot(target_ps, z, dims=1)
                next_a = torch.argmax(target_qs, dim=-1)
            target_p_unproj = select_at_indexes(next_a, target_ps)
            target_p_unproj = target_p_unproj.unsqueeze(1)
            target_p = (target_p_unproj * projection_coeffs).sum(-1)
        p = select_at_indexes(actions[index + 1].squeeze(-1), log_pred_ps)
        losses = -torch.sum(target_p * p, dim=-1)

        target_p = torch.clamp(target_p, EPS, 1)
        KL_div = torch.sum(target_p *
                           (torch.log(target_p) - p.detach()), dim=-1)
        KL_div = torch.clamp(KL_div, EPS, 1 / EPS)

        return losses, KL_div.detach()

    @torch.no_grad()
    def sample_goals(self, observation):
        proj_latents, _, latents = sample_goals(observation, self.model.encode_targets)
        if self.conv_goal:
            goals = latents.squeeze(0)
        else:
            goals = proj_latents.squeeze(0)

        goals = add_noise(goals, self.goal_noise_weight)
        goals = permute_goals(goals, self.goal_permute_prob)

        if 'gru' in self.model.transition_type:
            goals = self.model.renormalize_tensor(goals)
        else:
            goals = self.model.renormalize(goals)

        return goals

    def loss(self, samples, offline=False):
        if self.model.noisy:
            self.model.head.reset_noise()
            self.agent.target_model.head.reset_noise()

        observations = samples.all_observation.to(self.agent.device)
        actions = samples.all_action.to(self.agent.device)
        rewards = samples.all_reward.to(self.agent.device)
        dones = samples.done.to(self.agent.device)
        done_ns = samples.done_n.to(self.agent.device)
        nonterminals = 1. - torch.sign(torch.cumsum(dones, 0)).float()
        nonterminals_n = 1. - torch.sign(torch.cumsum(done_ns, 0)).float()

        if self.goal_weight > 0:
            goals = self.sample_goals(observations[1:self.goal_window])
        else:
            goals = None

        log_pred_ps, goal_log_pred_ps, spr_loss, latents, proj_latents, diversity, inverse_model_loss, bc_preds, \
        kl_loss, embed_cos, proj_cos = self.agent(observations, actions, rewards, goals, train=True)  # [B,A,P]

        # we always detach target representation gradient for goal loss
        latents = latents.detach()
        proj_latents = proj_latents.detach()

        batch_size = latents.shape[0]

        if self.rl_weight > 0:
            returns = samples.return_.to(self.agent.device)
            rl_loss, KL = self.rl_loss(log_pred_ps[:self.batch_size], observations[:, :self.batch_size],
                                       None, actions[:, :self.batch_size],
                                       rewards[:, :self.batch_size], nonterminals_n[:, :self.batch_size],
                                       returns[:, :self.batch_size], 0, self.n_step_return)
        else:
            rl_loss = torch.zeros((batch_size)).to(self.agent.device)
            KL = torch.zeros_like(rl_loss)

        if self.bc_weight > 0:
            log_pred_actions = F.log_softmax(bc_preds, -1)
            targets = actions[1]
            bc_loss = F.nll_loss(log_pred_actions, targets)
        else:
            bc_loss = torch.zeros_like(rl_loss)

        if self.goal_weight > 0:
            if self.conv_goal:
                goal_latents = latents[:, :self.goal_n_step + 1]
            else:
                goal_latents = proj_latents[:, :self.goal_n_step + 1]
            goal_returns = calculate_returns(goal_latents,
                                             goals,
                                             self.goal_distance,
                                             self.discount,
                                             nonterminals[:self.goal_n_step],
                                             distance_scale=5.,
                                             reward_scale=self.goal_reward_scale,
                                             all_to_all=self.goal_all_to_all)

            if self.goal_all_to_all:
                goal_nonterminals = nonterminals[None, None, self.goal_n_step]
                goal_nonterminals = goal_nonterminals.expand(-1, goal_nonterminals.shape[-1], -1).flatten(-2, -1)
                goal_actions = actions[:, None, :].expand(-1, actions.shape[-1], -1).flatten(-2, -1)
            else:
                goal_nonterminals = nonterminals[None, self.goal_n_step]
                goal_actions = actions

            goal_loss, goal_KL = self.rl_loss(goal_log_pred_ps, observations,
                                              goals, goal_actions,
                                              rewards,
                                              goal_nonterminals,
                                              goal_returns,
                                              0,
                                              self.goal_n_step)

            if self.goal_all_to_all:
                bs = actions.shape[1]
                goal_loss = goal_loss.view(bs, bs).mean(1)
                goal_KL = goal_KL.view(bs, bs).mean(1)

        else:
            goal_loss = goal_KL = torch.zeros((batch_size)).to(self.agent.device)

        if self.model.ssl_obj == 'byol':
            spr_loss = spr_loss * nonterminals[:self.jumps + 1]
            if self.jumps > 0:
                model_spr_loss = spr_loss[self.model.warmup:].mean(0) if self.model.use_latent else spr_loss[1:].mean(0)
                t0_spr_loss = spr_loss[0]
            else:
                t0_spr_loss = spr_loss[0]
                model_spr_loss = torch.zeros_like(spr_loss)
            t0_spr_loss = t0_spr_loss
            model_spr_loss = model_spr_loss
        elif self.model.ssl_obj == 'barlow' or self.model.ssl_obj == 'vicreg':
            model_spr_loss = spr_loss.mean()
            t0_spr_loss = torch.tensor(0.).to(self.agent.device)

        if not offline and self.prioritized_replay:  # revisit for barlow
            weights = samples.is_weights.to(rl_loss.device)
            t0_spr_loss = t0_spr_loss * weights
            model_spr_loss = model_spr_loss * weights
            goal_loss = goal_loss * weights
            bc_loss = bc_loss * weights
            inverse_model_loss = inverse_model_loss * weights
            rl_loss = rl_loss * weights

        return rl_loss.mean(), KL, \
               goal_loss.mean(), \
               t0_spr_loss.mean(), \
               model_spr_loss.mean(), \
               diversity, \
               inverse_model_loss.mean(), \
               bc_loss.mean(), \
               goal_KL, \
               kl_loss, \
               embed_cos, proj_cos
