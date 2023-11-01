# FedNIDS: A Federated Learning Framework for Packet-based Network Intrusion Detection System

This repository contains the code for the project "FedNIDS: A Federated Learning Framework for Packet-based Network Intrusion Detection System".

## Dataset
https://usf.box.com/s/3hj01f8hslsdazoqmr03tgmbxqk4oroy

## Paper abstract

**<p align="center">Figure 1: Proposed FedNIDS Framework.</p>**
<p align="center">
  <img src="https://github.com/quocnh/Fed_NIDS/blob/main/fig_fednids.png" width="90%"/>
</p>

Network intrusion detection systems (NIDS) play a critical role in discerning between benign and malicious network traffic. Deep neural networks (DNNs), anchored on large and diverse data sets, exhibit promise in enhancing the detection accuracy of NIDS by capturing intricate network traffic patterns. However, safeguarding distributed computer networks against emerging cyber threats is increasingly challenging. Despite the abundance of diverse network data, decentralization persists due to data privacy and security concerns. This confers an asymmetric advantage to adversaries, as distributed networks face the formidable task of securely and efficiently sharing non-independently and identically distributed data to counter cyber-attacks. To address this, we propose the Federated NIDS, a novel two-stage framework that combines the power of federated learning and DNNs. It aims to enhance the detection accuracy of known attacks, robustness and resilience to novel attack patterns, and privacy preservation, using packet-level granular data. In the first stage, a global DNN model is collaboratively trained on distributed data, and the second stage adapts it to novel attack patterns. Our experiments on real-world intrusion data sets demonstrate the effectiveness of FedNIDS by achieving an average F1 score of 0.97 across distributed networks and identifying novel attacks within four rounds of communication.

## Project Structure
```
|---- Dataset
|---- Source
|---- README.md
```

## Reproduce Exp1:
```
|---- nids_fedavg.ipynb
|---- nids_centralized.ipynb
|---- nids_fedprox.ipynb
```
## Reproduce Exp2:
```
|---- nids_fedavg_2017_2018_silo0more80.ipynb
|---- nids_fedavg_2017_2018_silo1more80.ipynb
|---- nids_fedavg_2017_2018_silo2more80.ipynb
|---- nids_fedavg_2017_2018_silo3more80.ipynb
```


## Citation
If you find this repository useful in your research, please cite the following articles as: 

```
@article{
  title={FedNIDS: A Federated Learning Framework for Packet-based Network Intrusion Detection System},
  author={Quoc H. Nguyen∗, Soumyadeep Hore∗, Ankit Shah, Trung Le, and Nathaniel D. Bastian},
  journal={},
  year={2023}
}

```
