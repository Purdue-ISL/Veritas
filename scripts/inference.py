import argparse
import os
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", required=True)
    parser.add_argument("--trained_model", required=True)
    args = parser.parse_args()
    return args

args = parse_arguments()

config_file = os.path.join(args.input_directory, 'test_config.yaml')
with open(config_file, 'r') as file:
    test_config = yaml.safe_load(file)

cmd = f"python3 -u transform.py \
        --suffix {test_config['suffix']}-{args.trained_model} \
        --dataset {args.input_directory} \
        --transform {args.input_directory}/full.json \
        --seed {test_config['seed']} --device {test_config['device']} \
        {'--jit' if test_config['jit'] else ''} \
        --resume {args.trained_model} \
        --num-random-samples {test_config['num_random_samples']} \
        --num-sample-seconds {test_config['num_sample_seconds']}"

print(cmd)
# os.system(cmd)