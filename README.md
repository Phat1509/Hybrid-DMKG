# Multimodal Knowledge Graph Project

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2512.00881)
[![Conference](https://img.shields.io/badge/AAAI-2026-blue)](#)

A research project for **multimodal knowledge graph construction** that supports **relation extraction**, **entity linking**, and **multimodal reasoning** by integrating **large language models (LLMs)** and **vision-language models (VLMs)**.

## 📌 Introduction

This project presents a **multimodal knowledge graph framework** for extracting structured knowledge from both **text** and **images**. It combines relation-aware reasoning with external knowledge base grounding to support downstream tasks such as:

- **Relation Extraction**
- **Entity Linking**
- **Multimodal Reasoning**
- **Knowledge Graph Construction**

The framework is designed to leverage both LLMs and VLMs for improved understanding across modalities.

---

## ✨ Highlights

- Build knowledge graphs from **textual** and **visual** information
- Perform **relation extraction** from natural language queries
- Link entities to external knowledge bases such as **Wikidata**
- Support **multimodal reasoning** with LLMs and VLMs
- Modular project structure for experimentation and extension

---

## 📄 Paper

- **arXiv**: [2512.00881](https://arxiv.org/pdf/2512.00881)

## 🎓 Publication

Accepted at **AAAI 2026**

---

## 📁 Project Structure

```bash
.
├── archive/                  # backups and experimental scripts
├── datasets/                 # embeddings, images, triples
├── en_core_web_md/           # spaCy model
├── hugging_cache/            # model APIs and cache files
├── prompts/                  # LLM prompts
├── RAE_graph/                # graph modules
├── test/                     # experiments and evaluation scripts
├── train/                    # training code and configurations for Relation Extraction
├── wikidata_tools/           # data processing
├── kedkg_requirements.txt    # project dependencies
└── my_llava_requirements.txt # LLaVA-related dependencies
```

---

## ⚙️ Installation

Install the required dependencies:

```bash
pip install -r kedkg_requirements.txt
pip install -r my_llava_requirements.txt
python -m spacy download en_core_web_md
```

You may also manually install some common packages if needed:

```bash
pip install torch transformers spacy sentence-transformers flask Pillow
```

---

## ⚠️ Notes

This project involves external knowledge sources such as **Wikidata** (and other wiki-related resources).  
If you are running the project in **mainland China**, you may need proper network access to ensure these resources are reachable.

---

## 🏋️ Training and Checkpoints

The `train/` directory contains the **training scripts, settings, and related configurations for the Relation Extraction module**.

The trained checkpoints obtained from `train/`, as well as some files related to `archive/`, can be found in the shared Baidu Netdisk folder below:

- **Link**: https://pan.baidu.com/s/18c5bNWeoye5o_2s21LGwOg?pwd=jgx4 
- **Extraction code**: `jgx4`
After downloading, place the checkpoint in the appropriate directory, for example:

```bash
../train/results_relation/checkpoint-4140
```

---

## 🚀 Quick Start

Optional VLM services:

```bash
python blip2opt_api.py --port 5006
python minigpt4_api.py --port 5005
python llava_api.py --port 5008
```

Run experiments:

```bash
cd test
python dynamic_exp_LLAMA.py --model_name llama2 --divide_modal llama2
```

---

## 🧪 Relation Extraction Example

```python
from extraction_knowlege import RelationExtractor

extractor = RelationExtractor(
    relation_model_path="../train/results_relation/checkpoint-4140",
    spacy_model_path="../en_core_web_md"
)
```

---

## 🔍 Supported Tasks

This project currently supports the following tasks:

- **Knowledge Graph Construction**
- **Relation Extraction**
- **Entity Linking**
- **Multimodal Entity Reasoning**
- **Integration with External Knowledge Bases**

---

## 📖 Citation

If you find this project useful in your research, please consider citing:

```bibtex
@inproceedings{yuan2026hybrid,
  title={Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing},
  author={Yuan, Li and Huang, Qingfei and Zhu, Bingshan and Cai, Yi and Huang, Qingbao and Zheng, Changmeng and Deng, Zikun and Wang, Tao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={33},
  pages={28032--28040},
  year={2026}
}
```

---

## 📬 Contact

For questions, discussions, or collaborations, please open an issue or contact the authors.

---

## 📜 License

This project is intended for academic and research purposes. Please add an appropriate license before open-source release.
