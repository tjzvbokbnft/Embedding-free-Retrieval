<p align="center">
  <img src="https://img.shields.io/badge/arXiv-coming--soon-red.svg?style=flat-square" alt="arXiv coming soon">
</p>


# Embedding-free-Retrieval

This repository presents a lightweight, embedding-free retrieval-augmented generation (RAG) framework that replaces vector search with lexical overlap and iterative keyword refinement. It leverages LLM reasoning instead of static embeddings to enable fast and accurate long-context retrieval.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Framework](#Framework)
- [Performance Analysis](#Performance-Analysis)
- [🔧 Installation & Quick Start](#-installation--quick-start)
- [📄 How to Get Results on NovelQA and Marathon](#-how-to-get-results-on-novelqa-and-marathon)


## Features

- **Embedding-Free Retrieval**: Eliminates the reliance on embedding models and dense retrievers.

- **Fast Response & Deployment**: Minimal offline preparation and indexing time enables fast responses and rapid deployment.

- **Lightweight Reasoning**: Utilizes LLMs’ native lexical and reasoning abilities—no need to construct graph or tree databases.

- **Strong Performance**: Outperforms baselines on long-context QA benchmarks such as NovelQA and Marathon.

## Project Structure

- `src/`: Core implementation of the embedding-free RAG framework.
- `reproduce/`: Scripts for reproducing experiments.
- `demo.py`: A runnable demo to test the pipeline.
- `requirements.txt`: List of dependencies.
- `.gitignore`: Standard ignored files.
- `Readme.md`: Project documentation.

## Framework
![framework](framework/framework1.png)
## Performance Analysis

### QA Retrieval Accuracy
![QA Retrieval Accuracy](performance/ret.png)

### Average Query Time by Context Length and Query Volume
![Average Query Time](performance/Time%203D.jpg)

### Total Time Consumption by Context Length
![Total Time Consumption](performance/Total%20time.jpg)


## 🔧 Installation & Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/tjzvbokbnft/Embedding-free-Retrieval.git
cd Embedding-free-Retrieval

#string_noise --> optional (To use this package, you must manually add it to your site-packages directory.
#If you don't need it, simply delete the related code.)
pip install -r requirements.txt

#play with demo
#config your txt path in demo.py
#config other parameters in scr/local_config.py
python demo.py

# Reproduce on NovelQA benchmark
python reproduce/test_novelqa.py

# Reproduce on Marathon benchmark
python reproduce/test_marathon.py
```
### 2.Basic Usages


## 📄 How to Get Results on NovelQA and Marathon

### 📊 Benchmark & Evaluation

To evaluate the performance of this framework on **NovelQA** and **Marathon**, please follow the official benchmarking instructions:

#### 📘 NovelQA
- [GitHub Repository](https://github.com/NovelQA/novelqa.github.io)
- [Official leaderboard](https://novelqa.github.io/)
  

#### 🏃 Marathon
- [GitHub Repository](https://github.com/Hambaobao/Marathon)
- [Submission](https://openbenchmark.online/marathon/)

##License
- MIT 

##Citation
- If Embedding-free Retrieval assists in your research, please cite it as follows:

