import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
import torch.distributions as td

from rlpyt.models.utils import update_state_dict
from rlpyt.utils.tensor import select_at_indexes
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from src.utils import (
    count_parameters,
    get_augmentation,
    from_categorical,
    find_weight_norm,
    update_state_dict_compat,
    off_diagonal,
)
from src.networks import *

import copy
import math


class SPRCatDqnModel(torch.nn.Module):
    """2D conlutional network feeding into MLP with ``n_atoms`` outputs
    per action, representing a discrete probability distribution of Q-values."""

    def __init__(
        self,
        image_shape,
        output_size,
        n_atoms,
        dueling,
        jumps,
        spr,
        augmentation,
        target_augmentation,
        eval_augmentation,
        dynamics_blocks,
        norm_type,
        noisy_nets,
        aug_prob,
        projection,
        imagesize,
        dqn_hidden_size,
        momentum_tau,
        renormalize,
        q_l1_type,
        predictor,
        rl,
        bc,
        kl,
        bc_from_values,
        goal_rl,
        goal_n_step,
        noisy_nets_std,
        residual_tm,
        inverse_model,
        goal_conditioning_type,
        transition_type,
        gru_input_size,
        gru_proj_size,
        gru_in_dropout,
        gru_out_dropout,
        ln_ratio,
        latent_dists,
        latent_dist_size,
        latent_proj_size,
        kl_balance,
        barlow_balance,
        free_nats,
        latent_merger,
        transition_layer_norm,
        transition_batch_norm,
        resblock="inverted",
        expand_ratio=2,
        freeze_encoder=False,
        share_l1=False,
        cnn_scale_factor=1,
        blocks_per_group=3,
        ln_for_rl_head=False,
        state_dict=None,
        conv_goal=True,
        goal_all_to_all=False,
        load_head_to=1,
        load_compat_mode=False,
        probe="prior",
        probe_jumps=[],
        probe_task="reward",
        probe_model="linear",
        warmup=0,
        input_bn=False,
        renormalize_type="ln_nt",
        ssl_obj="byol",
        barlow_lambd=0.0051,
        game="",
        use_ema=False,
        joint_embedding=False,
        projection_dim=1024,
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.game = game

        self.noisy = noisy_nets
        self.aug_prob = aug_prob
        self.projection_type = projection

        self.dqn_hidden_size = dqn_hidden_size

        if resblock == "inverted":
            resblock = InvertedResidual
        else:
            resblock = Residual

        self.transforms = get_augmentation(augmentation, imagesize)
        self.target_transforms = get_augmentation(target_augmentation, imagesize)
        self.eval_transforms = get_augmentation(eval_augmentation, imagesize)

        self.dueling = dueling
        f, c = image_shape[:2]
        in_channels = np.prod(image_shape[:2])

        self.input_bn = nn.BatchNorm2d(1, affine=False) if input_bn else None

        self.conv = nn.Sequential(
            ResnetCNN(
                in_channels,
                depths=[int(32 * cnn_scale_factor), int(64 * cnn_scale_factor), int(64 * cnn_scale_factor)],
                strides=[3, 2, 2],
                norm_type=norm_type,
                blocks_per_group=blocks_per_group,
                resblock=InvertedResidual,
                expand_ratio=expand_ratio,
            )
        )

        fake_input = torch.zeros(1, f * c, imagesize, imagesize)
        fake_output = self.conv(fake_input)
        fake_output = fake_output[-1] if isinstance(fake_output, list) else fake_output

        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1] * fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))

        self.renormalize = init_normalization(self.hidden_size, renormalize)

        self.jumps = jumps
        self.rl = rl
        self.bc = bc
        self.kl = kl
        self.bc_from_values = bc_from_values
        self.goal_n_step = goal_n_step
        self.use_spr = spr
        self.ssl_obj = ssl_obj
        self.target_augmentation = target_augmentation
        self.eval_augmentation = eval_augmentation
        self.num_actions = output_size

        self.head = GoalConditionedDuelingHead(
            self.hidden_size,
            output_size,
            hidden_size=self.dqn_hidden_size,
            pixels=self.pixels,
            noisy=self.noisy,
            conv_goals=conv_goal,
            goal_all_to_all=goal_all_to_all,
            share_l1=share_l1,
            n_atoms=n_atoms,
            ln_for_dqn=ln_for_rl_head,
            conditioning_type=goal_conditioning_type,
            std_init=noisy_nets_std,
        )

        self.transition_type = transition_type
        self.ln_ratio = ln_ratio
        self.latent_dists = latent_dists
        self.latent_dist_size = latent_dist_size
        self.use_latent = True if self.transition_type == "gru" else False
        self.kl_balance = kl_balance
        self.barlow_balance = barlow_balance
        self.free_nats = free_nats
        self.gru_proj_size = gru_proj_size
        repr_size = self.pixels * self.hidden_size

        if transition_type == "gru":
            self.dynamics_model = GRUModel(
                input_size=gru_input_size,
                proj_size=gru_proj_size,
                num_layers=1,
                num_actions=self.num_actions,
                stoch_size=self.latent_dists * self.latent_dist_size,
                dropout=gru_in_dropout,
                nonlinearity=nn.ELU,
                use_ln=transition_layer_norm,
                use_bn=transition_batch_norm,
            )

            self.prior_net = nn.Sequential(
                nn.Linear(gru_proj_size, latent_proj_size),
                nn.ELU(),
                nn.Linear(latent_proj_size, latent_dists * latent_dist_size),
            )

            self.posterior_net = nn.Sequential(
                nn.Linear(repr_size + gru_proj_size, latent_proj_size),
                nn.ELU(),
                nn.Linear(latent_proj_size, latent_dists * latent_dist_size),
            )
            if latent_merger == "linear":
                self.latent_merger = nn.Sequential(
                    nn.Linear(latent_dists * latent_dist_size + gru_proj_size, repr_size)
                )
            else:
                self.latent_merger = nn.Sequential(
                    nn.Linear(latent_dists * latent_dist_size + gru_proj_size, 600), nn.ELU(), nn.Linear(600, repr_size)
                )
        elif transition_type == "gru_det":
            self.dynamics_model = GRUModelDet(
                input_size=gru_input_size,
                repr_size=repr_size,
                proj_size=gru_proj_size,
                num_layers=1,
                num_actions=self.num_actions,
                dropout=gru_in_dropout,
                nonlinearity=nn.ELU,
            )
            self.gru_proj_out = nn.Sequential(
                nn.Linear(gru_proj_size, repr_size),
                nn.ELU(),
                nn.Dropout(gru_out_dropout),
            )
        else:
            self.dynamics_model = ConvDet(
                channels=self.hidden_size,
                num_actions=output_size,
                hidden_size=self.hidden_size,
                blocks=dynamics_blocks,
                norm_type=norm_type,
                resblock=resblock,
                expand_ratio=expand_ratio,
                renormalize=self.renormalize,
                residual=residual_tm,
            )

        self.renormalize_type = renormalize_type
        if self.renormalize_type == "ln":
            self.renormalize_layer = nn.LayerNorm(repr_size)
        elif self.renormalize_type == "bn":
            self.renormalize_layer = nn.BatchNorm1d(repr_size)
        elif self.renormalize_type == "bn_nt":
            self.renormalize_layer = nn.BatchNorm1d(repr_size, affine=False)
        elif self.renormalize_type == "ln_nt":
            self.renormalize_layer = nn.LayerNorm(repr_size, elementwise_affine=False)

        self.momentum_tau = momentum_tau
        self.use_ema = use_ema
        self.joint_embedding = joint_embedding

        if self.projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.Flatten(-3, -1),
                nn.Linear(self.pixels * self.hidden_size, 512),
                TransposedBN1D(512),
                nn.ReLU(),
                nn.Linear(512, 256),
            )
            self.target_projection = self.projection
            projection_size = 256
        elif self.projection_type == "q_l1":
            if goal_rl:
                layers = [self.head.goal_linears[0], self.head.goal_linears[2]]
            else:
                layers = [self.head.rl_linears[0], self.head.rl_linears[2]]
            self.projection = QL1Head(layers, dueling=dueling, type=q_l1_type)
            projection_size = self.projection.out_features
        elif self.projection_type == "linear":
            self.projection = nn.Sequential(nn.Linear(self.pixels * self.hidden_size, projection_dim))
            projection_size = projection_dim
        else:
            projection_size = self.pixels * self.hidden_size

        if self.joint_embedding:
            self.target_projection = self.projection
            self.target_encoder = self.conv
            self.target_renormalize_layer = self.renormalize_layer
        else:
            self.target_projection = self.projection
            self.target_projection = copy.deepcopy(self.target_projection)
            self.target_encoder = copy.deepcopy(self.conv)
            self.target_renormalize_layer = copy.deepcopy(self.renormalize_layer)
            for param in (
                list(self.target_encoder.parameters())
                + list(self.target_projection.parameters())
                + list(self.target_renormalize_layer.parameters())
            ):
                param.requires_grad = False

        if self.bc and not self.bc_from_values:
            self.bc_head = nn.Sequential(nn.ReLU(), nn.Linear(projection_size, output_size))

        if self.ssl_obj == "barlow":
            self.barlow_bn = nn.BatchNorm1d(projection_dim, affine=False)
            self.barlow_lambd = barlow_lambd

        # Gotta initialize this no matter what or the state dict won't load
        if predictor == "mlp":
            self.predictor = nn.Sequential(
                nn.Linear(projection_size, projection_size * 2),
                TransposedBN1D(projection_size * 2),
                nn.ReLU(),
                nn.Linear(projection_size * 2, projection_size),
            )
        elif predictor == "linear":
            self.predictor = nn.Sequential(
                nn.Linear(projection_size, projection_size),
            )
        elif predictor == "none":
            self.predictor = nn.Identity()

        self.use_inverse_model = inverse_model
        # Gotta initialize this no matter what or the state dict won't load
        self.inverse_model = InverseModelHead(
            projection_size,
            output_size,
        )

        print(
            "Initialized model with {} parameters; CNN has {}.".format(
                count_parameters(self), count_parameters(self.conv)
            )
        )
        print("Initialized CNN weight norm is {}".format(find_weight_norm(self.conv.parameters()).item()))

        if state_dict is not None:
            if load_compat_mode:
                state_dict = update_state_dict_compat(state_dict, self.state_dict())

            self.load_state_dict(state_dict)

            print("Loaded CNN weight norm is {}".format(find_weight_norm(self.conv.parameters()).item()))
            if rl:
                self.head.copy_base_params(up_to=load_head_to)
                if self.noisy:
                    self.head.reset_noise_params()

        # we initiate probe weights after model-load to ensure we always start with untrained probe weights
        self.probe = probe
        self.probe_jumps = probe_jumps
        self.warmup = warmup

        if self.probe and self.probe is not None:
            if probe_task == "reward":
                probe_out_dim = 1
            elif probe_task == "next_action":
                probe_out_dim = output_size
            else:
                raise NotImplementedError

            if probe_model == "linear":
                predictor = nn.Linear(fake_output.reshape(-1).shape[0], probe_out_dim)
            else:
                predictor = nn.Sequential(
                    nn.Linear(fake_output.reshape(-1).shape[0], 300), nn.ReLU(), nn.Linear(300, probe_out_dim)
                )

            self.init_reward_predictor_ft = copy.deepcopy(predictor)
            self.reward_predictor_ft = copy.deepcopy(predictor)
            self.embed_reward_predictor_ft = copy.deepcopy(predictor)

        self.frozen_encoder = freeze_encoder
        if self.frozen_encoder:
            self.freeze_encoder()

    def set_sampling(self, sampling):
        if self.noisy:
            self.head.set_sampling(sampling)

    def freeze_encoder(self):
        print("Freezing CNN")
        for param in self.conv.parameters():
            param.requires_grad = False

    def byol_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2.0, dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2.0, dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
        return loss

    def do_byol_loss(self, pred_latents, targets, observation):
        pred_latents = self.predictor(pred_latents)

        targets = targets.view(-1, observation.shape[1], self.jumps + 1, targets.shape[-1]).transpose(1, 2)
        latents = pred_latents.view(-1, observation.shape[1], self.jumps + 1, pred_latents.shape[-1]).transpose(1, 2)

        byol_loss = self.byol_loss(latents, targets).view(-1, observation.shape[1])  # split to batch, jumps

        return byol_loss

    def barlow_loss(self, feats1, feats2):
        z1 = self.barlow_bn(feats1)
        z2 = self.barlow_bn(feats2)
        cor = torch.mm(z1.T, z2)
        cor.div_(z1.shape[0])
        on_diag = torch.diagonal(cor).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(cor).pow_(2).sum()
        loss = on_diag + self.barlow_lambd * off_diag
        return loss

    def do_barlow_loss(self, feats1, feats2):
        feats1 = feats1.flatten(0, 1)
        feats2 = feats2.flatten(0, 1)

        # batch implementation
        if self.barlow_balance == 0.5:
            return self.barlow_loss(feats1, feats2)
        else:
            left_barlow_loss = self.barlow_loss(feats1, feats2.detach())
            right_barlow_loss = self.barlow_loss(feats1.detach(), feats2)
            return self.barlow_balance * left_barlow_loss + (1 - self.barlow_balance) * right_barlow_loss

    def do_spr_loss(self, pred_latents, targets, observation):
        if self.ssl_obj == "byol":
            loss = self.do_byol_loss(pred_latents.flatten(0, 1), targets.flatten(0, 1), observation)
        elif self.ssl_obj == "barlow":
            loss = self.do_barlow_loss(pred_latents, targets)
        else:
            raise NotImplementedError

        return loss

    @torch.no_grad()
    def calculate_diversity(self, global_latents, observation):
        global_latents = global_latents.view(observation.shape[1], self.jumps + 1, global_latents.shape[-1])[:, 0]
        # shape is jumps, bs, dim
        global_latents = F.normalize(global_latents, p=2.0, dim=-1, eps=1e-3)
        cos_sim = torch.matmul(global_latents, global_latents.transpose(0, 1))
        mask = 1 - (torch.eye(cos_sim.shape[0], device=cos_sim.device, dtype=torch.float))

        cos_sim = cos_sim * mask
        offset = cos_sim.shape[-1] / (cos_sim.shape[-1] - 1)
        cos_sim = cos_sim.mean() * offset

        return cos_sim

    def apply_transforms(self, transforms, image):
        for transform in transforms:
            image = maybe_transform(image, transform, p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, transforms, augment=False):
        images = images.float() / 255.0 if images.dtype == torch.uint8 else images
        if augment:
            flat_images = images.reshape(-1, *images.shape[-3:])
            processed_images = self.apply_transforms(transforms, flat_images)
            images = processed_images.view(*images.shape[:-3], *processed_images.shape[1:])

        if self.input_bn is not None:
            if len(images.shape) == 4:
                bs, stack, w, h = images.shape
                images = images.view(bs * stack, 1, w, h)
                images = self.input_bn(images).view(bs, stack, w, h)
            else:
                bs, jumps, stack, w, h = images.shape
                images = images.view(bs * jumps * stack, 1, w, h)
                images = self.input_bn(images).view(bs, jumps, stack, w, h)

        return images

    def split_stem_model_params(self):
        stem_params = list(self.conv.parameters()) + list(self.head.parameters())
        model_params = self.dynamics_model.parameters()

        return stem_params, model_params

    def sort_params(self, params_dict):
        return [params_dict[k] for k in sorted(params_dict.keys())]

    def list_params(self):
        all_parameters = {k: v for k, v in self.named_parameters()}
        conv_params = {k: v for k, v in all_parameters.items() if k.startswith("conv")}
        dynamics_model_params = {k: v for k, v in all_parameters.items() if k.startswith("dynamics_model")}

        q_l1_params = {
            k: v
            for k, v in all_parameters.items()
            if (
                k.startswith("head.goal_value.0")
                or k.startswith("head.goal_advantage.0")
                or k.startswith("head.rl_value.0")
                or k.startswith("head.rl_advantage.0")
            )
        }

        other_params = {
            k: v
            for k, v in all_parameters.items()
            if not (
                k.startswith("target")
                or k in conv_params.keys()
                or k in dynamics_model_params.keys()
                or k in q_l1_params.keys()
            )
        }

        return (
            self.sort_params(conv_params),
            self.sort_params(dynamics_model_params),
            self.sort_params(q_l1_params),
            self.sort_params(other_params),
        )

    def stem_forward(self, img):
        """Returns the normalized output of convolutional layers."""
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        with torch.no_grad() if self.frozen_encoder else nullcontext():
            conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
            conv_out = conv_out[-1] if isinstance(conv_out, list) else conv_out
            if "gru" in self.transition_type:
                conv_out_renorm = self.renormalize_tensor(conv_out)
            else:
                conv_out_renorm = self.renormalize(conv_out)

        return conv_out_renorm, conv_out

    def head_forward(self, conv_out, goal=None, logits=False):
        lead_dim, T, B, img_shape = infer_leading_dims(conv_out, 3)
        p = self.head(conv_out, goal)

        if logits:
            p = F.log_softmax(p, dim=-1)
        else:
            p = F.softmax(p, dim=-1)

        p = restore_leading_dims(p, lead_dim, T, B)
        return p

    @torch.no_grad()
    def encode_targets(self, images, project=True):
        images = self.transform(images, self.transforms, True)

        encoder = self.target_encoder
        projection = self.target_projection

        conv_out = encoder(images.flatten(0, 1))
        conv_out = conv_out[-1] if isinstance(conv_out, list) else conv_out

        if "gru" in self.transition_type:
            latents = self.renormalize_tensor(conv_out)
        else:
            latents = self.renormalize(conv_out)

        conv_out = conv_out.view(images.shape[0], images.shape[1], -1)

        if project:
            proj_latents = projection(latents)
            proj_latents = proj_latents.view(images.shape[0], images.shape[1], -1)

            return proj_latents, conv_out, latents.view(images.shape[0], images.shape[1], -1)
        else:
            return latents.view(images.shape[0], images.shape[1], -1)

    def propagate_forward(self, observations, prev_action, jumps, eval=False):
        bs = observations.shape[0]
        pred_latents = []
        priors = []
        posteriors = []
        det_states = []
        stochs = []
        embeds = []
        reprs = []

        # initialize the initial deterministic component to 0s
        latent = torch.zeros(bs, self.gru_proj_size).to(observations.device)

        stoch = torch.zeros(bs, self.latent_dists * self.latent_dist_size).to(observations.device)

        for j in range(jumps + 1):
            input_obs = observations[:, j, :]
            if eval:
                input_obs = self.transform(input_obs, self.eval_transforms)
            else:
                input_obs = self.transform(input_obs, self.transforms, augment=True)
            embed, _ = self.stem_forward(input_obs)

            latent, posterior_logits, prior_logits, stoch = self.step(
                latent, prev_action[j], embed.flatten(1, -1), stoch, eval=eval
            )

            embeds.append(embed)
            priors.append(prior_logits)
            posteriors.append(posterior_logits)
            stochs.append(stoch)

            repr_renorm, repr, latent = latent
            pred_latents.append(repr_renorm)
            reprs.append(repr)
            det_states.append(latent)

        return embeds, pred_latents, det_states, priors, posteriors, stochs, reprs

    def imagine_forward(self, latent, stoch, actions, jumps):
        pred_latents = []
        priors = []
        det_states = []

        for j in range(jumps):
            latent = self.dynamics_model(latent, actions[j], stoch)
            prior_logits = self.prior_net(latent)
            stoch = self.sample_discrete(prior_logits, argmax=True)

            next_repr = self.latent_merger(torch.cat((latent, stoch), dim=1))
            next_repr = self.renormalize_tensor(next_repr)

            priors.append(prior_logits)
            pred_latents.append(next_repr)
            det_states.append(latent)

        return pred_latents

    def imagine_forward_det(self, latent, actions, jumps):
        pred_latents = []
        reprs = []

        if self.transition_type == "gru_det":
            latent = latent.flatten(1, -1)

        for j in range(jumps):
            latent = self.step_det(latent, actions[j])

            if len(latent) == 3:
                repr_renorm, repr, latent = latent
                pred_latents.append(repr_renorm)
            else:
                latent, repr = latent
                pred_latents.append(latent.flatten(-3, -1))
            reprs.append(repr)

        return pred_latents, reprs

    def forward(self, observation, prev_action, prev_reward, goal=None, train=False, eval=False):
        """
        For convenience reasons with DistributedDataParallel the forward method
        has been split into two cases, one for training and one for eval.
        """
        if train:
            if self.use_latent:
                to_encode = max(self.jumps + 1, self.goal_n_step)
                all_input_obs = observation[:to_encode].transpose(0, 1).flatten(2, 3)

                latents, pred_latents, det_states, priors, posteriors, _, reprs = self.propagate_forward(
                    all_input_obs, prev_action, self.jumps
                )
                latent = latents[0]
            else:
                input_obs = observation[0].flatten(1, 2)
                input_obs = self.transform(input_obs, self.transforms, augment=True)
                latent, conv_out = self.stem_forward(input_obs)

                pred_latents = [latent.flatten(1, -1)]
                pred_latents_jumps, reprs = self.imagine_forward_det(latent, prev_action[1:], self.jumps)
                pred_latents += pred_latents_jumps

                posteriors = []
                priors = []

            if self.rl or self.bc_from_values:
                log_pred_ps = self.head_forward(latent, goal=None, logits=True)
            else:
                log_pred_ps = None

            if goal is not None:
                goal_log_pred_ps = self.head_forward(latent, goal=goal, logits=True)
            else:
                goal_log_pred_ps = None

            # compute target
            with nullcontext() if self.joint_embedding else torch.no_grad():
                to_encode = max(self.jumps + 1, self.goal_n_step)
                target_observations = observation[:to_encode].transpose(0, 1).flatten(2, 3)
                target_latents = []

                for j in range(self.jumps + 1):
                    target_obs = target_observations[:, j, :]
                    target_obs = self.transform(target_obs, self.target_transforms, True)
                    target_latent = self.target_encoder(target_obs)
                    target_latent = target_latent[-1] if isinstance(target_latent, list) else target_latent
                    target_latents.append(target_latent)

                for j in range(self.jumps + 1):
                    if "gru" in self.transition_type:
                        target_latents[j] = self.renormalize_tensor(target_latents[j], target=True)
                    else:
                        target_latents[j] = self.renormalize(target_latents[j])

                target_latents = torch.stack(target_latents, dim=1).flatten(2, 4)
                target_proj = self.target_projection(target_latents)

            pred_latents = torch.stack(pred_latents, 1)
            proj_latents = self.projection(pred_latents)

            if self.use_spr:
                spr_loss = self.do_spr_loss(proj_latents, target_proj, observation)
            else:
                spr_loss = torch.zeros((self.jumps + 1, observation.shape[1]), device=latents.device)

            # measure cosine similarities
            with torch.no_grad():
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                embed_cos = torch.mean(
                    cos(pred_latents[:, 1:, :].flatten(0, 1), target_latents[:, 1:, :].flatten(0, 1))
                )
                proj_cos = torch.mean(cos(proj_latents[:, 1:, :].flatten(0, 1), target_proj[:, 1:, :].flatten(0, 1)))

            latent_kl_loss = self.latent_kl_loss(posteriors, priors)

            if self.bc:
                if self.bc_from_values:
                    bc_preds = from_categorical(log_pred_ps.exp(), limit=10, logits=False)

                if self.bc and not self.bc_from_values:
                    bc_preds = self.bc_head(proj_latents[:, 0])
            else:
                bc_preds = None

            if self.use_inverse_model:
                # we always detach target grad for inverse loss
                stack = torch.cat([proj_latents[:, :-1], target_proj.detach().view(*proj_latents.shape)[:, 1:]], -1)
                pred_actions = self.inverse_model(stack.flatten(0, 1))
                pred_actions = pred_actions.view(stack.shape[0], stack.shape[1], *pred_actions.shape[1:])
                pred_actions = pred_actions.transpose(0, 1)
                # correct impl
                inv_model_loss = F.cross_entropy(
                    pred_actions.flatten(0, 1), prev_action[1 : self.jumps + 1].flatten(0, 1), reduction="none"
                )
                inv_model_loss = inv_model_loss.view(*pred_actions.shape[:-1]).mean(0)
            else:
                inv_model_loss = torch.zeros_like(spr_loss).mean(0)

            diversity = self.calculate_diversity(proj_latents, observation)

            if self.use_ema:
                update_state_dict(self.target_encoder, self.conv.state_dict(), self.momentum_tau)
                update_state_dict(self.target_projection, self.projection.state_dict(), self.momentum_tau)
                update_state_dict(self.target_renormalize_layer, self.renormalize_layer.state_dict(), self.momentum_tau)

            return (
                log_pred_ps,
                goal_log_pred_ps,
                spr_loss,
                target_latents,
                target_proj,
                diversity,
                inv_model_loss,
                bc_preds,
                latent_kl_loss,
                embed_cos,
                proj_cos,
            )
        else:
            observation = observation.flatten(-4, -3)

            transforms = self.eval_transforms if eval else self.target_transforms

            img = self.transform(observation, transforms, len(transforms) > 0)

            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

            conv_out = self.conv(img.view(T * B, *img_shape))
            conv_out = conv_out[-1] if isinstance(conv_out, list) else conv_out

            if "gru" in self.transition_type:
                conv_out = self.renormalize_tensor(conv_out)
            else:
                conv_out = self.renormalize(conv_out)
            p = self.head(conv_out, goal)

            p = F.softmax(p, dim=-1)

            p = restore_leading_dims(p, lead_dim, T, B)

            return p

    def select_action(self, obs):
        if self.bc_from_values or not self.bc:
            value = self.forward(obs, None, None, train=False, eval=True)
            value = from_categorical(value, logits=False, limit=10)
        else:
            observation = obs.flatten(-4, -3)
            img = self.transform(observation, self.eval_transforms, len(self.eval_transforms) > 0)
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
            conv_out = self.conv(img.view(T * B, *img_shape))
            conv_out = conv_out[-1] if isinstance(conv_out, list) else conv_out

            if "gru" in self.transition_type:
                conv_out = self.renormalize_tensor(conv_out)
            else:
                conv_out = self.renormalize(conv_out)
            proj = self.projection(conv_out)
            value = self.bc_head(proj)
            value = restore_leading_dims(value, lead_dim, T, B)
        return value

    def step_det(self, state, action):
        next_state = self.dynamics_model(state, action)
        if "gru" in self.transition_type:
            next_repr = self.gru_proj_out(next_state)
            next_repr_renorm = self.renormalize_tensor(next_repr)
            next_state = (next_repr_renorm, next_repr, next_state)
        else:
            next_repr = next_state
            next_repr_renorm = self.renormalize(next_repr)
            next_state = (next_repr_renorm, next_repr)

        return next_state

    def step(self, state, action, next_embed, stoch, eval=False):
        next_state = self.dynamics_model(state, action, stoch)

        posterior_logits = self.posterior_net(torch.cat((next_embed, next_state), dim=1))
        next_stoch = self.sample_discrete(posterior_logits, argmax=eval)
        next_repr = self.latent_merger(torch.cat((next_state, next_stoch), dim=1))

        prior_logits = self.prior_net(next_state)

        next_repr_renorm = self.renormalize_tensor(next_repr)
        next_state = (next_repr_renorm, next_repr, next_state)

        return next_state, posterior_logits, prior_logits, next_stoch

    def sample_discrete(self, logits, argmax=False):
        # straight through gradient
        bs = logits.shape[0]
        logits = logits.view(bs, self.latent_dists, self.latent_dist_size)

        m = torch.distributions.OneHotCategorical(logits=logits)

        if argmax:
            argmax_inds = torch.argmax(m.probs, dim=-1)
            samples = F.one_hot(argmax_inds, num_classes=self.latent_dist_size).to(logits.device, dtype=logits.dtype)
        else:
            samples = m.sample()
            samples = samples + m.probs - m.probs.detach()

        samples = samples.view(bs, self.latent_dists * self.latent_dist_size)
        return samples

    def latent_kl_loss(self, posteriors, priors):
        if not self.use_latent:
            return torch.tensor(0.0)

        posteriors = torch.cat(posteriors[1:])
        priors = torch.cat(priors[1:])

        bs = posteriors.shape[0]
        posteriors = posteriors.view(bs, self.latent_dists, self.latent_dist_size)
        priors = priors.view(bs, self.latent_dists, self.latent_dist_size)

        left_kl = torch.mean(
            torch.distributions.kl.kl_divergence(
                td.Independent(td.OneHotCategoricalStraightThrough(logits=posteriors), 1),
                td.Independent(td.OneHotCategoricalStraightThrough(logits=priors.detach()), 1),
            )
        )

        right_kl = torch.mean(
            torch.distributions.kl.kl_divergence(
                td.Independent(td.OneHotCategoricalStraightThrough(logits=posteriors.detach()), 1),
                td.Independent(td.OneHotCategoricalStraightThrough(logits=priors), 1),
            )
        )

        left_kl = max(left_kl, torch.tensor(self.free_nats, dtype=torch.float).to(left_kl.device))
        right_kl = max(right_kl, torch.tensor(self.free_nats, dtype=torch.float).to(right_kl.device))

        kl_loss = (1 - self.kl_balance) * left_kl + self.kl_balance * right_kl

        return kl_loss

    def renormalize_tensor(self, tensor, first_dim=1, target=False):
        flat = len(tensor.shape) < 4

        renormalize_layer = self.target_renormalize_layer if target else self.renormalize_layer

        if flat:
            flat_tensor = tensor
        else:
            if first_dim < 0:
                first_dim = len(tensor.shape) + first_dim
            flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)

        if self.renormalize_type == "ln" or self.renormalize_type == "bn":
            flat_tensor = renormalize_layer(flat_tensor)
        elif self.renormalize_type == "ln_nt":
            flat_tensor = renormalize_layer(flat_tensor) * 0.1115 + 0.2
        elif self.renormalize_type == "bn_nt":
            flat_tensor = renormalize_layer(flat_tensor) * 0.09
        else:
            raise NotImplementedError

        if flat:
            return flat_tensor
        else:
            return flat_tensor.view(*tensor.shape)


class QL1Head(nn.Module):
    def __init__(self, layers, dueling=False, type=""):
        super().__init__()
        self.noisy = "noisy" in type
        self.dueling = dueling
        self.relu = "relu" in type

        self.encoders = nn.ModuleList(layers)
        self.out_features = sum([encoder.out_features for encoder in self.encoders])

    def forward(self, x):
        if len(x.shape) > 3:
            x = x.flatten(-3, -1)
        representations = []
        for encoder in self.encoders:
            encoder.noise_override = self.noisy
            representations.append(encoder(x))
            encoder.noise_override = None
        representation = torch.cat(representations, -1)
        if self.relu:
            representation = F.relu(representation)

        return representation


def maybe_transform(image, transform, p=0.8):
    processed_images = transform(image)

    if p >= 1:
        return processed_images
    else:
        mask = torch.rand((processed_images.shape[0], 1, 1, 1), device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * image
        return processed_images
