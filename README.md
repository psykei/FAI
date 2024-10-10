# FaUCI: Fairness Under Constrained Injection

## Description

This repo contains the experiments for the paper mentioned in the title.

## Requirements

The code is written in Python 3.10.0. To install the required packages, run:

```pip install -r requirements.txt```

All experiments were run on a MacBook Pro Apple M1 chip with 16GB of RAM.

## Organisation

The code is organised as follows:
- `dataset/` contains the dataset used in the experiments along with the code for its loading and preprocessing.
- `fairness/` contains the code for the fairness metrics of `our` method (FaUCI), `cho` and `jiang` methods.
- `analysis/` contains the code for executing and gathering of the results.
- `images/` contains the code to generate images (some of them are included in the paper).
- `configuration.py` contains the setup of the experiments.

Concerning Cho and Jiang methods, we used the code provided by the authors of the papers.
The code is available at the following links:
- Cho: https://proceedings.neurips.cc/paper_files/paper/2020/file/ac3870fcad1cfc367825cda0101eee62-Supplemental.zip
- Jiang: https://github.com/zhimengj0326/GDP

## Usage

To run the experiments, execute the following command:

```python fairness/our/__main__.py```

The results will be saved in `fairness/our/log` folder.
Similarly, for `cho` and `jiang` methods, run:

```python fairness/cho/__main__.py```

```python fairness/jiang/__main__.py```

To analise the results for each method, run:

```python analysis/our/__main__.py```

```python analysis/cho/__main__.py```

```python analysis/jiang/__main__.py```

To generate comparison images, run:

```python images/__main__.py```
