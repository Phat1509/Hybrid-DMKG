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
├── archive/           # backups and experimental scripts
├── datasets/          # embeddings, images, triples
├── en_core_web_md/    # spaCy model
├── hugging_cache/     # model APIs
├── prompts/           # LLM prompts
├── RAE_graph/         # graph modules
├── test/              # experiments
├── train/             # training code
└── wikidata_tools/    # data processing
```

---

## ⚙️ Installation

Install the required dependencies:

```bash
pip install torch transformers spacy sentence-transformers flask Pillow
python -m spacy download en_core_web_md
```

---

## 🚀 Quick Start

### 1. Start model APIs

```bash
cd hugging_cache
python llama2_api.py
```

Optional VLM services:

```bash
python blip2opt_api.py --port 5006
python minigpt4_api.py --port 5005
python llava_api.py --port 5008
```

### 2. Run experiments

```bash
cd test
python dynamic_exp_LLAMA.py --model_name llama2 --divide_modal llama2
```

---

## 🧪 Relation Extraction Example

```python
from extraction_knowlege import RelationExtractor

extractor = RelationExtractor(
    relation_model_path='../train/results_relation/checkpoint-4140',
    spacy_model_path='../en_core_web_md'
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
@article{yourname2026multimodal,
  title={Multimodal Knowledge Graph Construction via Relation-Aware Entity Reasoning},
  author={Your Name et al.},
  journal={AAAI},
  year={2026}
}
```

---

## 📬 Contact

For questions, discussions, or collaborations, please open an issue or contact the authors.

---

## 📜 License

This project is intended for academic and research purposes. Please add an appropriate license before open-source release.
