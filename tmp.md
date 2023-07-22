Veritas is an easy-to-interpret domain-specific ML model that relates the latent stochastic process (intrinsic bandwidth that the video session can achieve) to actual observations (download times), while exploiting counterfactual queries via abduction using the observed TCP states (e.g., congestion window) for blocking the cascading dependencies. 

## Pre-requisites:

The following set up has been tested on Ubuntu 22.04.
```
# Installing pip3
sudo apt update
sudo apt upgrade
sudo apt install python3-pip 
# Installing conda
    * https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
# Clone the repository.
cd VeritasML
conda create --name veritas
conda activate veritas
bash environment.sh
```
## Input to the Veritas (src/data/dataset/):
```
 input_directory/
    |_ video_session_streams/
    |_ ground_truth_capacity/
    |_ fhash.json
    |_ full.json
```
* video_session_streams directory consists of video session files, wherein each line in the video session file includes information about chunk payloads in a video session. The fields included in the file are: the start time (numpy.datetime64), end time (numpy.datetime64), size (KB), trans_time/download time (ms), cwnd (number), rtt (ms),rto (ms), ssthresh (number), last_snd (s), min_rtt (ms), delivery_rate (-).

* ground_truth_capacity directory: it contains the ground truth information about the underlying INB associated with a given video session. Each line in a ground truth capacity file includes the ground truth bandwidth (Mbps) and start time (numpy.datetime64) for that capacity. In case of emulation experiments, we would know the ground truth INB and Veritas aspires to match these values. In real world experiments, we do not know the ground truth capacity. In this case, these files serve as a reference and do not imply the INB associated with the video sessions. 

* full.json has a list with the names of the video session files.
* fhash.json has a hash value for each file in the ground_truth_capacity and video_session_streams directory.

## Domain-specific emission model (f):
Veritas has the flexibility to use custom functions for the emission models of Veritas’s High-order Embedded Hidden Markov Model (HoEHMM). We pass the emission functions in the fit.py and transform.py files. These functions use the fields described in the video_session_file (except download time) and the capacity values passed for abduction as inputs and return the estimated throughput. For reference, we have included a few emission functions in fit.py and transform.py.

## Training Veritas:
```
python3 fit.py -h 
usage: Fit by HMM Algorithms [-h] [--suffix SUFFIX] --dataset DATASET --train TRAIN --valid VALID --test TEST [--seed SEED] --device DEVICE [--jit] --initial
                             INITIAL --transition TRANSITION --emission EMISSION [--smooth SMOOTH] --num-epochs NUM_EPOCHS [--eq-eps EQ_EPS] --capacity-max
                             CAPACITY_MAX [--filter-capmax] --capacity-unit CAPACITY_UNIT [--capacity-min CAPACITY_MIN] --transition-unit TRANSITION_UNIT
                             --initeta INITETA --transeta TRANSETA --vareta VARETA --varinit VARINIT --varmax-head VARMAX_HEAD --varmax-rest VARMAX_REST
                             --head-by-time HEAD_BY_TIME --head-by-chunk HEAD_BY_CHUNK [--transextra TRANSEXTRA] [--include-beyond] [--support SUPPORT]

options:
  -h, --help            show this help message and exit
  --suffix SUFFIX       Saving title suffix.
  --dataset DATASET     Dataset directory.
  --train TRAIN         Training index definition.
  --valid VALID         Validation index definition.
  --test TEST           Training index definition.
  --seed SEED           Random seed.
  --device DEVICE       Computation device.
  --jit                 Enable JIT.
  --initial INITIAL     Initial model.
  --transition TRANSITION
                        Transition model.
  --emission EMISSION   Emission model.
  --smooth SMOOTH       Transition model smoother.
  --num-epochs NUM_EPOCHS
                        Number of epochs.
  --eq-eps EQ_EPS       Equality tolerance.
  --capacity-max CAPACITY_MAX
                        Maximum capacity.
  --filter-capmax       Filter maximum capacity in dataset.
  --capacity-unit CAPACITY_UNIT
                        Capacity unit.
  --capacity-min CAPACITY_MIN
                        Minimum capacity.
  --transition-unit TRANSITION_UNIT
                        Transition unit time.
  --initeta INITETA     Weight for emission initial distribution update.
  --transeta TRANSETA   Weight for emission transition matrix update.
  --vareta VARETA       Weight for emission variance update.
  --varinit VARINIT     Constant initialization for emission variance update.
  --varmax-head VARMAX_HEAD
                        Maximum for heading emission variance update.
  --varmax-rest VARMAX_REST
                        Maximum for heading emission variance update.
  --head-by-time HEAD_BY_TIME
                        Heading variance time (second).
  --head-by-chunk HEAD_BY_CHUNK
                        Heading variance chunks.
  --transextra TRANSEXTRA
                        Estimation transition extra arguments.
  --include-beyond      Include target states which is invalid in estimation transition construction.
  --support SUPPORT     Path to external supporting data for estimation.
```
```
Example: python3 -u fit.py --suffix Controlled-GT-Cubic-BBA-LMH-gaussian.asym-v --dataset src/data/datasets/Controlled-GT-Cubic-BBA-LMH --train src/data/datasets/Controlled-GT-Cubic-BBA-LMH/full.json --valid src/data/datasets/Controlled-GT-Cubic-BBA-LMH/full.json --test src/data/datasets/Controlled-GT-Cubic-BBA-LMH/full.json --seed 42 --device cpu --jit --initial generic --transition gaussian.asym --emission v --num-epochs 25 --capacity-max 8.0 --capacity-unit 0.5 --transition-unit 5.0 --initeta 0 --transeta 0.1 --vareta 0.0001 --varinit --varmax-head --varmax-rest --head-by-time 5.0 --head-by-chunk 5 --transextra 5 --include-beyond --smooth 0.05
```
The trained model is stored at logs/fit/ directory.

## Inference using Veritas:
```
python3 transform.py -h 
usage: Transform by HMM Algorithms [-h] [--suffix SUFFIX] --dataset DATASET --transform TRANSFORM [--seed SEED] --device DEVICE [--jit] --resume RESUME
                                   [--num-random-samples NUM_RANDOM_SAMPLES] --num-sample-seconds NUM_SAMPLE_SECONDS [--disable-step-size] [--disable-dense-bar]
                                   [--disable-true-capacity]

options:
  -h, --help            show this help message and exit
  --suffix SUFFIX       Saving title suffix.
  --dataset DATASET     Dataset directory.
  --transform TRANSFORM
                        Transformation index definition.
  --seed SEED           Random seed.
  --device DEVICE       Computation device.
  --jit                 Enable JIT.
  --resume RESUME       Resume from given log.
  --num-random-samples NUM_RANDOM_SAMPLES
                        Number of random samples.
  --num-sample-seconds NUM_SAMPLE_SECONDS
                        Number of seconds (can be float) per sample. This argument assumes training time unit is second.
  --disable-step-size   Disable step size rendering.
  --disable-dense-bar   Disable density bar rendering.
  --disable-true-capacity
                        Disable true capacity rendering.
```
```
Example: python3 -u transform.py --suffix Controlled-GT-Cubic-BBA-LMH<= --dataset src/data/datasets/Controlled-GT-Cubic-BBA-LMH --transform src/data/datasets/Controlled-GT-Cubic-BBA-LMH/full.json --seed 42 --device cpu --jit --resume logs/fit/: --num-random-samples 3 --num-sample-seconds 300
```
The output INB traces estimated by Veritas along with figures comparing throughput, INB and Veritas estimated INB will be stored at src/logs/transform/ directory.

## Example:
We share the emulation datasets from our experiments in the paper:

Steps to run:
1. Fit the model for a given dataset. 
    * bash scripts/fit.sh
2. Transform the dataset and get the capacity traces.
    * bash scripts/transform.sh

