import argparse
import os
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", required=True)
    args = parser.parse_args()
    return args

args = parse_arguments()

config_file = os.path.join(args.input_directory, 'train_config.yaml')
with open(config_file, 'r') as file:
    train_config = yaml.safe_load(file)

cmd = f"python3 -u fit.py \
        --suffix {train_config['suffix']}-{train_config['transition']}-{train_config['emission']} \
        --dataset {args.input_directory} \
        --train {args.input_directory}/full.json --valid {args.input_directory}/full.json --test {args.input_directory}/full.json \
        --seed {train_config['seed']} --device {train_config['device']} \
        {'--jit' if train_config['jit'] else ''} \
        --initial {train_config['initial']} --transition {train_config['transition']} \
        --emission {train_config['emission']} \
        --num-epochs {train_config['num_epochs']} \
        --capacity-max {train_config['capacity_max']} --capacity-unit {train_config['capacity_unit']} \
        --transition-unit {train_config['transition_unit']} \
        --initeta {train_config['initeta']} --transeta {train_config['transeta']} \
        --vareta {train_config['vareta']} --varinit {train_config['varinit']} \
        --varmax-head {train_config['varmax_head']} \
        --varmax-rest {train_config['varmax_rest']} \
        --head-by-time {train_config['head_by_time']} --head-by-chunk {train_config['head_by_chunk']} \
        --transextra  {train_config['trans_extra']} \
        {'--include-beyond' if train_config['include_beyond'] else ''} \
        --smooth {train_config['smooth']}"

print(cmd)
os.system(cmd)