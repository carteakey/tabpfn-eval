# TabPFN Replicated

The [TabPFN](https://github.com/automl/TabPFN/) is a neural network that learned to do tabular data prediction. 

## Installation
Need Python 3.9 or below
```bash
conda create --name tabpfn python=3.9
conda activate tabpfn
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
conda activate tabpfn-automl
```

```bash
pip install autosklearn==0.14.5
pip install autogluon==0.4.0
```

## Getting Started

```shell
src
├── __init__.py                         # initializing package
├── evals
    ├── datasets.py                     # selecting the datasets from OpenML
    ├── eval_automl_models.py           # evaluating AutoML models
    ├── eval_baseline_models.py         # evaluating baseline models
    ├── eval_baseline_tabpfn.py         # evaluating baseline TabPFN model
    ├── eval_gbdt_models.py             # evaluating GBDT models
    ├── eval_modified_tabpfn.py          # evaluating modified TabPFN model
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
#### Running the demo code 

See `src/tabpfn_demo.py` for scikit-learn interface and a basic classification task.

```python
python -m src.tabpfn_vs_xgb_demo
# Accuracy (TabPFN) = 97.872%
# Accuracy (XGBoost) = 96.809%
```

###### Running the trained model

Download the retrained/modified model - `prior_diff_real_checkpointkc_n_0_epoch_100.cpkt` and save it to `src/tabpfn/models_diff`
```
cd ./src/tabpfn/models_diff
wget https://github.com/carteakey/tabpfn-eval/raw/main/src/tabpfn/models_diff/prior_diff_real_checkpointkc_n_0_epoch_100.cpkt
```

- Please change BASE_DIR in `src/__init__.py`

###### Generating datasets
```
(tabpfn) kchauhan@kpc:~/repos/tabpfn-eval$ python3 -m src.evals.datasets
```
##### Running evals
```
python -m src.evals.eval_baseline_models_wo_hyperopt
```




## Evaluation

### TabPFN (original)
- No of datasets: 247
- Mean ROC: 0.846
- Mean Cross Entropy: nan
- Mean Accuracy: 0.803
- Mean Prediction Time: 0.233s
 
### TabPFN (retrained/modified) - trained for just 5 hours.
- No of datasets: 247
- Mean ROC: 0.837
- Mean Cross Entropy: 0.451
- Mean Accuracy: 0.792
- Mean Prediction Time: 0.388s

See `training_log.txt`


### About TabPFN usage
TabPFN is different from other methods you might know for tabular classification. Here, we list some tips and tricks that might help you understand how to use it best.

* Do not preprocess inputs to TabPFN. TabPFN pre-processes inputs internally. It applies a z-score normalization (`x-train_x.mean()/train_x.std()`) per feature (fitted on the training set) and log-scales outliers heuristically. Finally, TabPFN applies a PowerTransform to all features for every second ensemble member. Pre-processing is important for the TabPFN to make sure that the real-world dataset lies in the distribution of the synthetic datasets seen during training. So to get the best results, do not apply a PowerTransformation to the inputs.

* TabPFN expects scalar values only (you need to encode categoricals as integers e.g. with OrdinalEncoder). It works best on data that does not contain any categorical or NaN data.

* TabPFN ensembles multiple input encodings per default. It feeds different index rotations of the features and labels to the model per ensemble member. You can control the ensembling with `TabPFNClassifier(...,N_ensemble_configurations=?)`

* TabPFN does not use any statistics from the test set. That means predicting each test example one-by-one will yield the same result as feeding the whole test set together.

* TabPFN is differentiable in principle, only the pre-processing is not and relies on numpy.
