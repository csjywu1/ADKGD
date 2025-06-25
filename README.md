## ADKGD: Anomaly Detection in Knowledge Graphs with Dual-Channel Training


### Overview

**ADKGD** is a novel anomaly detection algorithm for Knowledge Graphs (KGs) using a dual-channel learning framework. It leverages both entity-view and triplet-view representation learning, combined with cross-layer information aggregation and a KL-divergence-based consistency loss, to robustly identify anomalous triplets (i.e., errors or noise) in KGs.

ADKGD **outperforms state-of-the-art methods** on several benchmark datasets including WN18RR, FB15K, and NELL-995, and is suitable for ensuring the reliability of KGs for downstream applications such as question answering and recommendation systems.


### Installation

```bash
git clone https://github.com/csjywu1/ADKGD.git
cd ADKGD
```
### Datasets

Supported datasets:

* FB15K-237
* WN18RR
* NELL-995
* Kinship
* YAGO
* KG20C

The complete datasets and code could be downloaded from https://pan.baidu.com/s/17snkQlOTNtD4IsYau3pxqg?pwd=gkhc 提取码: gkhc 


### Citation

If you use ADKGD in your research or for your project, please cite our paper:

bibtex
@article{wu2025adkgd,
  title={ADKGD: Anomaly Detection in Knowledge Graphs with Dual-Channel Training},
  author={Jiayang Wu and Wensheng Gan and Jiahao Zhang and Philip S. Yu},
  journal={ACM Transactions on Asian and Low-Resource Language Information Processing},
  year={2025},
  url={https://arxiv.org/abs/2501.07078},
}


### Contact

For questions, feel free to open an issue or email Jiayang Wu ([csjywu1@gmail.com](mailto:csjywu1@gmail.com)).
