import os
import json
import hashlib
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", required=True)
    args = parser.parse_args()
    return args

args = parse_arguments()

fhash = args.input_directory + '/fhash.json'

files = os.listdir(args.input_directory + '/ground_truth_capacity')

res = {}
res['ground_truth_capacity'] = {}
res['video_session_streams'] = {}

for file_name in files:
    # make a hash object
    h = hashlib.sha256()

    # open file for reading in binary mode
    with open(args.input_directory + '/ground_truth_capacity' + '/' + file_name, 'rb') as file:

        # loop till the end of the file
        chunk = 0
        while chunk != b'':
            # read only 1024 bytes at a time
            chunk = file.read(1024)
            h.update(chunk)

    # return the hex representation of digest

    res['ground_truth_capacity'][file_name] = h.hexdigest()

    g = hashlib.sha256()

    # open file for reading in binary mode
    with open(args.input_directory + '/video_session_streams' + '/' + file_name, 'rb') as file:

        # loop till the end of the file
        chunk = 0
        while chunk != b'':
            # read only 1024 bytes at a time
            chunk = file.read(1024)
            g.update(chunk)

    # return the hex representation of digest

    res['video_session_streams'][file_name] = g.hexdigest()

with open(fhash, 'w') as convert_file:
    convert_file.write(json.dumps(res))
