# CapAware — Capacity-Aware Uplink Bandwidth Prediction for Cellular Networks

As remotely controlled and autonomous vehicles become widely available, the demand for high Quality of Service over cellular networks for their remote control and monitoring is becoming increasingly important. Accurate prediction of available uplink bandwidth is essential to mitigate bandwidth fluctuations and avoid impacting real-time applications, ensuring reliable and low-latency video streams. In particular, bandwidth overpredictions lead to packet losses, retransmissions, and significant latency increases, especially during network handovers, as network buffers fill up. Prior bandwidth prediction approaches lower absolute or relative errors but fail to address the impacts of overpredictions and the associated latency spikes.

This paper introduces CapAware, a bandwidth prediction approach explicitly designed to minimize capacity violations (i.e., overpredictions) and reduce latency spikes during network handovers for uplink streams. It utilizes an efficient neural network architecture with an integrated handover prediction mechanism and a learnable capacity-aware loss function. CapAware predicts network handovers with a 92.4% F1 score and improves efficiency by 24.4% using its custom loss function with predicted handover information. Compared to deep-learning baselines, CapAware improves network efficiency (i.e., utilization-to-capacity violation ratio) by 4.7% and 34.9% on 5G SA datasets.

It was presented at the 50th [IEEE Conference on Local Computer Networks (LCN)](https://www.ieeelcn.org/) and won the [Best Paper Award](https://www.ieeelcn.org/Annals_Awards.html).

This project is licensed under the terms of the Creative Commons Attribution 4.0 International License.

## Usage:

1. Clone this Git repository to your local machine using the following command:

```
git clone https://github.com/ds-kiel/CapAware.git
```

2. Create a virtual Python environment (e.g. using pyenv)

```
curl https://pyenv.run | bash
pyenv install 3.13
pyenv virtualenv 3.13 CapAware
```

3. Install the necessary Python packages by running:

```
cd CapAware
pip install -r requirements.txt -U
```

4. Analyze and train handover and bandwidth prediction models
- Train the handover prediction model
- Use the trained model to do inference on the bandwidth prediction data to create probabilities needed to train the bandwidth prediction model
- Train the bandwidth prediction model using the new combined dataset

## Citation
B. Denizer and O. Landsiedel, "CapAware: Capacity-Aware Uplink Bandwidth Prediction for Cellular Networks," 2025 IEEE 50th Conference on Local Computer Networks (LCN), Sydney, Australia, 2025, pp. 1-9
```
@INPROCEEDINGS{11146351,
  author={Denizer, Birkan and Landsiedel, Olaf},
  booktitle={2025 IEEE 50th Conference on Local Computer Networks (LCN)}, 
  title={CapAware: Capacity-Aware Uplink Bandwidth Prediction for Cellular Networks}, 
  year={2025},
  volume={},
  number={},
  pages={1-9},
  keywords={Cellular networks;5G mobile communication;Bandwidth;Handover;Predictive models;Throughput;Real-time systems;Uplink;Standards;Remote control;Bandwidth prediction;handover prediction;capacity-aware;utilization;overprediction;5G},
  doi={10.1109/LCN65610.2025.11146351}}
```