#!/bin/bash
declare -A map=( ["pong"]="Pong" ["breakout"]="Breakout" ["up_n_down"]="UpNDown" ["kangaroo"]="Kangaroo" ["bank_heist"]="BankHeist" ["assault"]="Assault" ["boxing"]="Boxing" ["battle_zone"]="BattleZone" ["frostbite"]="Frostbite" ["crazy_climber"]="CrazyClimber" ["chopper_command"]="ChopperCommand" ["demon_attack"]="DemonAttack" ["alien"]="Alien" ["kung_fu_master"]="KungFuMaster" ["qbert"]="Qbert" ["ms_pacman"]="MsPacman" ["hero"]="Hero" ["seaquest"]="Seaquest" ["jamesbond"]="Jamesbond" ["amidar"]="Amidar" ["asterix"]="Asterix" ["private_eye"]="PrivateEye" ["gopher"]="Gopher" ["krull"]="Krull" ["freeway"]="Freeway" ["road_runner"]="RoadRunner" )
export game=$1
shift
export seed=$1
shift
export group=$1

python -m scripts.run \
  agent.model_kwargs.blocks_per_group=3 \
  agent.model_kwargs.cnn_scale_factor=1 \
  agent.model_kwargs.expand_ratio=2 \
  agent.model_kwargs.freeze_encoder=True \
  agent.model_kwargs.gru_input_size=250 \
  agent.model_kwargs.joint_embedding=False \
  agent.model_kwargs.noisy_nets=1 \
  agent.model_kwargs.ssl_obj=byol \
  agent.model_kwargs.transition_type=gru_det \
  algo.batch_size=64 \
  algo.encoder_lr=2e-06 \
  algo.goal_weight=0 \
  algo.inverse_model_weight=0 \
  algo.jumps=1 \
  algo.kl_weight=0.0 \
  algo.learning_rate=0.0001 \
  algo.q_l1_lr=0.0001 \
  algo.spr_weight=0 \
  do_online=True \
  env.game=${game} \
  env.repeat_action_probability=0.25 \
  group=${group} \
  group_add=finetune \
  model_load=checkpoints/${group}/${game}_1 \
  offline.runner.dataloader.games=[${map[${game}]}] \
  runner.n_steps=100000 \
  seed=${seed} \
  wandb.disable_log=True \
  wandb.project=SGI_online