# TabPFN Replicated

The [TabPFN](https://github.com/automl/TabPFN/) is a neural network that learned to do tabular data prediction. 

## Installation
Need Python 3.9 or below
```bash
conda create --name tabpfn python=3.9
```

Install all the libraries
```bash
pip install tabpfn[full]
# install additional requirements
pip install requirements.txt
```

To run the autogluon and autosklearn baseline, 
create a separate environment and install 

```bash
conda create --name tabpfn-automl python=3.9
```

```bash
pip install autosklearn==0.14.5
pip install autogluon==0.4.0
```

## Getting Started

```shell
src
├── __init__.py                         # initializing package
├── datasets.py                         # selecting the datasets from OpenML
├── eval_automl_models.py               # evaluating AutoML models
├── eval_baseline_models.py             # evaluating baseline models
├── eval_baseline_tabpfn.py             # evaluating baseline TabPFN model
├── eval_gbdt_models.py                 # evaluating GBDT models
├── eval_modified_tabpfn.py              # evaluating modified TabPFN model
├── data
│   ├── openml_baseline_tabpfn.csv      # baseline results for TabPFN
│   └── openml_modified_tabpfn.csv       # modified results for TabPFN
│   ├── openml_list.csv                 # list of selected datasets and attributes
├── models
│   └── tabpfn
│       ├── modified                      # modified / retrained TabPFN
│       │   ├── model_configs.py          ## model configurations  
│       │   ├── models_diff              ## model checkpoints 
│       │   ├── tabpfn_new_params.json   ## training parameters 
│       │   ├── train.py                 ## training script
│       │   ├── train_config.py           ## training configurations 
│       │   └── train_continue.py        ## continuing training from checkpoint

│       └── original                     # original TabPFN
│           ├── model_configs.py          ## model configurations  
│           ├── models_diff              ## model checkpoints 
│           ├── tabpfn_orig_params.json  ## training parameters 
│           ├── tabpfn_orig_params.py    ## extract original training params
│           ├── train.py                 ## training script
│           └── train_config.py           ## training configurations 
├── plots
│   ├── classifier_decision_boundary.py  # plot classifier decision boundary
│   ├── transformer_model_viz.py        # plot transformer model architecture
│   └── visualize_priors.py             # plot Priors
├── tabpfn_demo.py                      # TabPFN demo using scikit-learn interface
└── utils.py                            # utility functions
```
#### Running the code 

- Please change BASE_DIR in `src/__init__.py`

See `src/tabpfn_demo.py` for scikit-learn interface and a basic classification task.


``` python -m src.tabpfn_vs_xgb_demo
# Accuracy (TabPFN) = 97.872%
# Accuracy (XGBoost) = 96.809%
```



Download the retrained/modified model - `prior_diff_real_checkpointkc_n_0_epoch_100.cpkt` and save it to `src/models/modified/models_diff`

###### Generating datasets
```
(tabpfn) kchauhan@kpc:~/repos/mds-tmu-dl$ python3 -m src.evals.datasets
```




## Evaluation

### TabPFN (original)
- No of datasets: 247
- Mean ROC: 0.843
- Mean Cross Entropy: nan
- Mean Accuracy: 0.802
- Mean Prediction Time: 0.357s
 
### TabPFN (retrained/modified) - trained for just 5 hours.
- No of datasets: 247
- Mean ROC: 0.837
- Mean Cross Entropy: 0.451
- Mean Accuracy: 0.792
- Mean Prediction Time: 0.39s

See `training_log.txt`
