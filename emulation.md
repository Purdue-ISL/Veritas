## Emulation setup
We use the Puffer setup, with minor changes to access additional TCP variables, to run the video streaming emulation experiments. 
Puffer allows to run various ABR algorithms with control over deployment settings such as choice of ABR, TCP congestion control, 
video qualities, client buffer size, etc. We emulate FCC throughput traces to play a 5 minute pre-recorded video clip with bitrate 
ranging from 0.1 Mbps to 4 Mbps using Mahimahi.

- Step 1: Use the Puffer documentation to deploy Puffer: https://github.com/StanfordSNR/puffer/wiki/Documentation
- Step 2: Deployment (Setting A): convert the FCC throughput traces to mahimahi format to run emulation with setting A.
- Step 3: Use the video session streams from Setting A and:
  - Baseline: generate baseline traces and run emulation using them.
  - Veritas: use the video session streams to generate the Veritas samples and run emulation using them.
- Step 4: Ground Truth (Setting B): Use the FCC traces to run emulation with setting B.
- Step 5: Collection the quality metrics for Ground Truth, Baseline and Veritas and compare them.


