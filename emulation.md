We use an evaluation setup similar to the figure below. A video session in Setting A is emulated using a ground truth network bandwidth (INB) trace. The resulting logs are provided to the different schemes. Both Veritas and Baseline produce traces
inferring INB. A video session is emulated in Setting B with the traces inferred by these schemes as well as the original INB trace
to obtain the metrics predicted by these schemes, and ground truth. Veritas samples multiple inferred traces (five by default) for each video session, each of which are emulated.

<img width="350" alt="image" src="https://github.com/Purdue-ISL/Veritas/assets/19619070/65781240-dc13-40e9-a39e-5b56ead0aeb4">

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


