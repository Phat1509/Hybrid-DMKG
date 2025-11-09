# 🧠 Hybrid-DMKG  
### A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing  

[![Conference](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/Conferences/AAAI-26/)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)  

---

## 📢 News
> 🎉 **Our paper “Hybrid-DMKG” has been accepted by AAAI 2026!**  
> We will release more code and the full dataset soon. Stay tuned!

---

## 📖 Overview  

**Hybrid-DMKG** is a **hybrid reasoning framework** for **multimodal multihop question answering (MMQA)** over **dynamic multimodal knowledge graphs (DMKGs)**, designed to support **knowledge editing** in both textual and visual modalities.

---

### 🔬 Background  

**Multimodal Knowledge Editing (MKE)** extends traditional textual knowledge editing to multimodal settings involving both language and vision.  
However, existing MKE benchmarks mainly evaluate final answer correctness, overlooking the **quality of intermediate reasoning steps** and **robustness to visually rephrased inputs**.

To address these limitations, we introduce **MMQAKE**, the **first benchmark** for **multimodal multihop question answering with knowledge editing**, which evaluates:

1. The model’s ability to reason over **2–5-hop factual chains** that span both text and images, including performance at each intermediate step.  
2. **Robustness** to visually rephrased multimodal questions after knowledge updates.

Our evaluation shows that existing MKE methods struggle to **consistently update and reason** over multimodal reasoning chains following knowledge edits.

