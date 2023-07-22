import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", required=True)
    args = parser.parse_args()
    return args

args = parse_arguments()

files = os.listdir(args.input_directory + '/video_session_streams')

f = open(args.input_directory + '/full.json', 'w')
f.write('[\n')
for i in range(len(files) - 1):
    f.write('\t"' + files[i] + '",\n')

f.write('\t"' + files[-1] + '"\n]')
f.close()
