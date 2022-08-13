#!/bin/bash
games='Amidar Assault Asterix Boxing DemonAttack Frostbite Gopher Krull Seaquest'
ckpts='3 4 5'
files='action observation reward terminal'
export data_dir=$1

echo "Missing Files:"
for g in ${games[@]}; do
  for f in ${files[@]}; do
    for c in ${ckpts[@]}; do
      if [ ! -f "${data_dir}/${g}/${f}_${c}.gz" ]; then
        echo "${data_dir}/${g}/${f}_${c}.gz"
      fi;
    done;
    if [ ! -f "${data_dir}/${g}/${f}_2_1.gz" ]; then
      echo "${data_dir}/${g}/${f}_2_1.gz"
    fi;
    if [ ! -f "${data_dir}/${g}/${f}_2_50.gz" ]; then
      echo "${data_dir}/${g}/${f}_2_50.gz"
    fi;
  done;
done;

# https://stackoverflow.com/a/226724
echo "Do you wish to download missing files?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done

for g in ${games[@]}; do
  mkdir -p "${data_dir}/${g}"
  for f in ${files[@]}; do
    # Pretrain dataset
    for c in ${ckpts[@]}; do
      if [ ! -f "${data_dir}/${g}/${f}_${c}.gz" ]; then
        gsutil cp "gs://atari-replay-datasets/dqn/${g}/1/replay_logs/\$store\$_${f}_ckpt.${c}.gz" "${data_dir}/${g}/${f}_${c}.gz"
      fi;
    done;

    # Reward probing dataset
    if [ ! -f "${data_dir}/${g}/${f}_2_1.gz" ]; then
      gsutil cp "gs://atari-replay-datasets/dqn/${g}/2/replay_logs/\$store\$_${f}_ckpt.1.gz" "${data_dir}/${g}/${f}_2_1.gz"
    fi;

    # Expert action probing dataset
    if [ ! -f "${data_dir}/${g}/${f}_2_50.gz" ]; then
      gsutil cp "gs://atari-replay-datasets/dqn/${g}/2/replay_logs/\$store\$_${f}_ckpt.50.gz" "${data_dir}/${g}/${f}_2_50.gz"
    fi;

  done;
done;
