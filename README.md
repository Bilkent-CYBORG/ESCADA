# ESCADA: Efficient Safe and Context Aware Dose Allocation for Precision Medicine
Author: Ilker Demirel ([ilkerd1997@gmail.com](mailto:ilkerd1997@gmail.com))

The repository for the manuscript "[ESCADA: Efficient Safe and Context Aware Dose Allocation for Precision Medicine](https://arxiv.org/abs/2111.13415)", (NeurIPS 2022).

## Requirements

```setup
pip install -r requirements.txt
```

## UVA/PADOVA T1DM simulator

We use the open source implementation of the UVa/PADOVA simulator, which is available under MIT License at [simglucose](https://github.com/jxx123/simglucose). We obtained the simulator outputs for different patient, meal event, insulin intake triples used in the experiments and saved the results in the following directories for faster reproduction,

```
/experiments/calc_res/*
/experiments/calc_res_clinician_data/*
```

## Reproduction

Our experimental results are classified under three main categories: single meal event (SME) scenario, multiple meal event (MME) scenario, and comparison against a clinician. We have separate `__.py` files for each setting, and the experiments can be run for each scenario by running the corresponding scripts in the following directories,

```experiments
/experiments/SME/SME.py
/experiments/MME/MME.py
/experiments/clinician_comparison/clinician_comparison.py
```

## Plots

The necessary data to obtain the plots and the numerical results in the paper is already available in the following directories,

```results
/experiments/SME/ppbg
/experiments/SME/ppbg_tc
/experiments/MME/ppbg
/experiments/clinician_comparison/test_res
```

You can run the experiments as described before to replace these results. Finally, to obtain the plots and the numerical results in the table, use the following notebook,

```
/experiments/plot.ipynb
```
