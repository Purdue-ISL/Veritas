## Veritas: Answering Causal Queries from Video Streaming Traces

Veritas is an easy-to-interpret domain-specific ML model to tackle causal reasoning for video streaming.

Given data collected from real video streaming sessions, a video publisher may wish to answer "what-if" questions such as 
understanding the performance if a different Adaptive Bitrate (ABR) algorithm were used , or if a new video quality 
(e.g., an 8K resolution) were added to the ABR selection. Veritas helps to answer such questions, also known as causal reasoning, 
without requiring data collected through RCTs.For a given video session, it uses the observed data (chunk download times, chunk 
sizes, TCP states, etc.) to infer the latent Intrinisic Network Bandwidth (INB) during the session.  Once an INB sample is 
obtained, we can now directly evaluate the proposed changes, and return the answer to the what-if query. Further, rather 
than a single point estimate, Veritas provides a range of potential outcomes reflecting the inherent uncertainty in inferences 
that can be made from the data.

This artifact accompanies the paper: Chandan Bothra, Jianfei Gao, Sanjay Rao, and Bruno Ribeiro. Veritas: Answering Causal Queries from Video Streaming Traces, ACM SIGCOMM 2023. Please cite this paper if you use the artifact.

## Pre-requisites:

The following set up has been tested on Ubuntu 22.04.
```
# Installing pip3
sudo apt update
sudo apt upgrade
sudo apt-get install python3-pip 
# Installing conda
    * https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html [python3.10]
# Clone the repository.
cd VeritasML
conda create --name veritas
conda activate veritas
bash environment.sh
```

## Preparing dataset for Veritas:

For a given video sessions dataset collected by user, create an input directory <input_directory> with the following structure:
   ```
    input_directory/
    |_ video_session_streams/
    |_ ground_truth_capacity/
    |_ train_config.yaml
    |_ inference_config.yaml
    |_ full.json
    |_ fhash.json

   ```
 - **train_config.yaml**: It contains various parameters needed to train a dataset.
 - **inference_config.yaml**: It contains various parameters needed for inference using a dataset.

 - **video_session_streams**: It contains the observed data relevant to a video session such as download time, 
 chunk size, TCP states (when available), etc.
 - **ground_truth_capacity**: This is useful for evaluating the performance of Veritas by comparing the inferred values 
 with ground truth, and to plot figures. In emulation experiments, the INB is known and Veritas samples aspire to match 
 the INB. In real world data, we do not know the INB, hence we can make a best guess for the INB. Please note: 
 this data is not necessary for the functioning of Veritas.
 - **full.json**: It contains a list of the video session files to be used to for evaluation.
 - **fhash.json**: It contains hash value for each file in the video_session_streams and ground_truth_capacity directory.

More details are shared [below](#input-dataset-details). For reference, we have shared a [dataset](./src/data/datasets/Controlled-GT-Cubic-BBA-LMH) used in the paper, which contains the files and directories mentioned above.


## Using Veritas (Reproducing results from paper)
The following steps run Veritas for training and inference. We use the above dataset as input, but any user input directory with above defined structure can be used as an input. Please note the commands are run from the home directory, VeritasML. 

1. Training: The parameters (general, HMM, video sessions, etc.) from training configuration file in the input directory are used for training. The trained model is saved in the logs/fit/ directory with the name: <timestamp>:<suffix_in_the_config_file>.
```
python3 scripts/train.py --input_directory src/data/datasets/Controlled-GT-Cubic-BBA-LMH
```
2. Inference: The output model from training and the parameters (number of samples, duration of samples, etc.) from the inference configuration file in the input directory are used for inference of INB traces.
```
python3 scripts/inference.py --input_directory src/data/datasets/Controlled-GT-Cubic-BBA-LMH --trained_model <path_to_trained_model>
```
The location and contents of output directory look like:
```
logs/transform/<timestamp>:<suffix_in_the_config_file>
   |_ sample
      |_<session_1>
        |_ sample_full.csv
      |_<session_2>
        |_ sample_full.csv
      ...
```
Each sample_full.csv is of 'd/t' lines and contains 'n' comma separated values for the inferred INB for the given 
session. 'n' is the number of samples and 'd' is the duration of sampled INB passed in the inference configuration 
file, while 't' is the transition step time passed during training. Example of sample_full.csv (with n=3):
```
0,1,2
4.5,4.5,4.5
3.5,3.5,3.5
3.0,3.5,3.0
3.0,3.5,3.0
3.0,3.0,3.0
3.5,3.5,3.0
```
## Using the inferred INB traces
The INB traces (using the sample_full.csv) can be used for bandwidth emulation using tools such as Mahimahi. In the emulation environment, we can now directly evaluate the proposed changes (change of ABR, change of qualities, etc), and return the answer to the what-if queries.

## Domain-specific emission model (f):	   
Veritas has the flexibility to use custom functions for the emission models of Veritas’s High-order Embedded Hidden Markov Model (HoEHMM). We pass the emission functions in the fit.py and transform.py files. These functions use the fields described in the video_session_file (except download time) and 
possible capacity values for abduction as inputs and return the estimated throughput. For reference, we have included a few emission functions in fit.py and transform.py.

## Input dataset details:
- **video_session_streams**: Each line in the video session file includes information about chunk payloads in a video session. The fields included in the file are: the start time (numpy.datetime64), end time (numpy.datetime64), size (KB), trans_time/download time (ms), cwnd (number), rtt (ms),rto (ms), ssthresh (number), last_snd (s), min_rtt (ms), delivery_rate (-). Ex: [sample_file](./src/data/datasets/Controlled-GT-Cubic-BBA-LMH/video_session_streams/fake_trace_10013_http---edition.cnn.com_76823454_cbs_6).
- **ground_truth_capacity**: Each line in a ground truth capacity file includes the ground truth bandwidth (Mbps) and start time (numpy.datetime64) for that capacity. Ex: [sample_file](./src/data/datasets/Controlled-GT-Cubic-BBA-LMH/ground_truth_capacity/fake_trace_10013_http---edition.cnn.com_76823454_cbs_6).
- **full.json**: It contains a list of the video session files to be used to for evaluation. Ex: [sample_file](./src/data/datasets/Controlled-GT-Cubic-BBA-LMH/full.json). This file is used to identify the sessions used for training, validation and inference. In our case, we use all the sessions for training and again 
use all the sessions for inference. Thus, full.json includes the names of all the sessions in the video_session_streams directory. The [script](./scripts/get_full.py) can be used to generate this file.
```
python3 scripts/get_full.py --input_directory <input_directory>
```
- **fhash.json**: It contains hash value for each file in the video_session_streams and ground_truth_capacity directory. It is useful to uniquely identify the input files and helps in logging the results. Ex: [sample_file](./src/data/datasets/Controlled-GT-Cubic-BBA-LMH/fhash.json). The [script](./scripts/get_fhash.py) can be used to generate this file.
```
python3 scripts/get_fhash.py --input_directory <input_directory>
```
## Contact
Please contact cbothra@purdue.edu for any questions.
   
   
