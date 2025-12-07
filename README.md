# DeepMLF: Multimodal language model with learnable tokens for deep fusion in sentiment analysis

<div align="center">

[![📄 Paper](https://img.shields.io/badge/Paper-arXiv%3A2504.11082-blue)](https://arxiv.org/abs/2504.11082)

<img src="figs/deepmlf.jpg" width="400">
</div>

## Abstract

While multimodal fusion has been extensively studied in \emph{multimodal sentiment analysis} (MSA) due to rich cross-modal complementary information, the role of fusion depth and multimodal capacity allocation have not been fully explored. In this work, we introduce DeepMLF, a novel multimodal language model design with learnable tokens tailored toward deep fusion. DeepMLF leverages an audiovisual encoder and a pretrained decoder language model (LM) augmented with multimodal information across its layers. DeepMLF appends learnable tokens to the LM with two objectives: 1) to capture modality interactions in a controlled fashion, and 2) to preserve independent information propagation for each modality. Fusion tokens gather linguistic information via causal self-attention within the LM, and are then integrated with audiovisual information through cross-attention blocks, acting as dedicated multimodal capacity. This design enables progressive fusion across multiple layers, providing fusion depth. Our training recipe combines modality-specific losses and LM loss, with the decoder LM predicting ground truth polarity. We evaluate DeepMLF on three MSA benchmarks, demonstrating improved performance over state-of-the-art methods across different languages, dataset sizes, and modality imbalance scenarios. Our results confirm that deeper fusion leads to better performance, with optimal fusion depths (5-7) exceeding existing approaches. Analysis reveals that few (8-20) tokens achieve optimal performance, providing insights on ideal capacity allocation. Through comprehensive experiments, we examine fusion curriculum, demonstrate scalability to large language models, and analyze language distribution effects and regularization impact. Our findings position fusion depth, scalability, and dedicated multimodal capacity as primary factors for effective multimodal fusion.

## TL;DR

DeepMLF introduces a novel multimodal language model with learnable fusion tokens that enables deep fusion across 5-7 layers, achieving state-of-the-art performance on MSA benchmarks. The key insight is that deeper fusion with dedicated multimodal capacity (small token sets) significantly outperforms shallow fusion approaches while maintaining efficient computation.

## 🔥 Highlights

- **State-of-the-art performance** across three MSA benchmarks (MOSI, MOSEI, SIMS) covering different languages and dataset characteristics
- **Deep fusion architecture** with optimal fusion depths of 5-7 layers, significantly deeper than existing approaches (typically ≤3 layers)
- **Efficient dedicated multimodal capacity** using only 8-20 learnable fusion tokens for optimal performance
- **Novel MM Blocks** with gated cross-attention that maintains independent information flow for each modality
- **Scalable design** compatible with various language model backbones (GPT2-base to SmolLM2-1.7B)
- **Fusion Curriculum** demonstrating the importance of fusion curriculum (language + audiovisual first; multimodal then)

## ⚙️ Setup

Clone the repository:
```bash
git clone https://github.com/efthymisgeo/deepmlf.git
cd deepmlf
```

Create and activate the conda environment:
```bash
conda env create -f environment.yml -n deepmlf
conda activate deepmlf
```

## 🚀 Train your DeepMLF

To train DeepMLF on your dataset:

### Train the AV Encoder [Optional]

To reprodcue the results from the paper and in general to acquire better resulkts, we suggest to independently train the AV encoder

```bash
python experiments/regression/mult_base.py \
  -m bienc \
  -d <dataset-name> \
  -g 0 \
  --exp-name bienc-... \
  -c MMSA/config/regression/deepmlf/.. \
  --res-save-dir MMSA/results/... \
  -n 2 \
  -s 1990
```

### Train DeepMLF

```bash
python experiments/regression/mult_base.py \
  -m msalm \
  -d <dataset-name> \
  -g 0 \
  --exp-name deepmlf-mosei-large \
  -c MMSA/config/regression/deepmlf/.. \
  --res-save-dir MMSA/results/... \
  -n 2 \
  -s 1990
```

Example configurations for each dataset:

```bash
# MOSEI with GPT2-large
python experiments/regression/mult_base.py -m msalm -d mosei -g 0 --exp-name deepmlf-large-mosei -c MMSA/config/regression/deepmlf/mosei/large_best.json --res-save-dir MMSA/results/deepmlf -n 2 -s 1990 -s 1991

# MOSI with SmolLM2-1.7B
python experiments/regression/mult_base.py -m msalm -d mosi -g 0 --exp-name deepmlf-smol-mosi -c MMSA/config/regression/deepmlf/mosi/best.json --res-save-dir MMSA/results/deepmlf -n 2 -s 1990 -s 1991

# SIMS with GPT2-base
python experiments/regression/mult_base.py -m msalm -d sims -g 0 --exp-name deepmlf-base-sims -c MMSA/config/regression/deepmlf/sims/base_best.json --res-save-dir MMSA/results/deepmlf -n 2 -s 1990 -s 1991
```

## 📁 Structure

Inside the `MMSA` folder we can modify the following scripts to customize your experiments

- [`models/`](MMSA/models/) - Core model implementations, `MSALM.py` for DeepMLF
- [`trains/`](MMSA/trains/) - Training scripts and utilities, `MSALM.py` for DeepMLF
- [`data_loader.py`](MMSA/data_loader.py) - Data loading and preprocessing
- [`config/`](MMSA/config/) - Configuration files for different experiments

## 📄 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{georgiou2025deepmlf,
  title={DeepMLF: Multimodal language model with learnable tokens for deep fusion in sentiment analysis},
  author={Georgiou, Efthymios and Katsouros, Vassilis and Avrithis, Yannis and Potamianos, Alexandros},
  journal={arXiv preprint arXiv:2504.11082},
  year={2025}
}
```

## 📋 TODO

- [ ] Remove dead code
- [ ] Upload pre-extracted features for MOSI, MOSEI, and SIMS datasets

## 🙏 Acknowledgments

- [MMSA](https://github.com/thuiar/MMSA): standardized training and evaluation MSA framework
- [open_flamingo](https://github.com/mlfoundations/open_flamingo): model related code
- [nanoGPT](https://github.com/mlfoundations/open_flamingo): starting point for initial DeepMLF model

## 📧 Contact

For questions, collaborations, or issues, please contact me directly efthymios[dot]georgiou[at]unibe[dot]ch
