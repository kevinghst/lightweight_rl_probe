#!/bin/bash
declare -A map=( ["pong"]="Pong" ["breakout"]="Breakout" ["up_n_down"]="UpNDown" ["kangaroo"]="Kangaroo" ["bank_heist"]="BankHeist" ["assault"]="Assault" ["boxing"]="Boxing" ["battle_zone"]="BattleZone" ["frostbite"]="Frostbite" ["crazy_climber"]="CrazyClimber" ["chopper_command"]="ChopperCommand" ["demon_attack"]="DemonAttack" ["alien"]="Alien" ["kung_fu_master"]="KungFuMaster" ["qbert"]="Qbert" ["ms_pacman"]="MsPacman" ["hero"]="Hero" ["seaquest"]="Seaquest" ["jamesbond"]="Jamesbond" ["amidar"]="Amidar" ["asterix"]="Asterix" ["private_eye"]="PrivateEye" ["gopher"]="Gopher" ["krull"]="Krull" ["freeway"]="Freeway" ["road_runner"]="RoadRunner" )
export game=$1
shift
export seed=$1
shift
export group=$1
shift
export data_path=$1

python -m scripts.run \
  +offline.algo.goal_weight=1 \
  +offline.algo.inverse_model_weight=1 \
  +offline.algo.spr_weight=1.0 \
  agent.model_kwargs.blocks_per_group=3 \
  agent.model_kwargs.cnn_scale_factor=1 \
  agent.model_kwargs.expand_ratio=2 \
  agent.model_kwargs.gru_input_size=600 \
  agent.model_kwargs.kl_balance=0.95 \
  agent.model_kwargs.latent_dist_size=32 \
  agent.model_kwargs.latent_dists=32 \
  agent.model_kwargs.ssl_obj=byol \
  agent.model_kwargs.transition_type=gru \
  agent.model_kwargs.use_ema=True \
  algo.batch_size=64 \
  algo.jumps=10 \
  algo.kl_weight=0.1 \
  algo.learning_rate=0.0001 \
  algo.probe_condition=5 \
  algo.probe_jumps=[0] \
  algo.probe_task=reward \
  env.game=${game} \
  group=${group} \
  offline.runner.dataloader.checkpoints='[3,4,5]' \
  offline.runner.dataloader.data_path=${data_path} \
  offline.runner.dataloader.ft_ckpt=1 \
  offline.runner.dataloader.games=[${map[${game}]}] \
  offline.runner.dataloader.samples=500000 \
  offline.runner.epochs=20 \
  offline.runner.no_init_eval=False \
  offline.runner.save_every=23437 \
  offline_model_save=checkpoints/${group}/${game}_${seed}/ \
  runner.eval_only=True \
  seed=${seed} \
  wandb.disable_log=True \
  model_load=checkpoints/${group}/${game}_${seed}/
