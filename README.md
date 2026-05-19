# Structure-Based-Genetic-Programming-for-Regression
 implementing structure-based genetic program- ming to predict the electricity load for a specific time on a particular day given the data for the n previous days at the same time or alternatively the m previous values can be used

## Files
- NB: csv data file should be stored in a folder called "data" 
- gp_load_forecasting.py: main script for experiments
- data_loader.py: CSV loading, lagged supervised dataset creation, time split
- gp_tree.py: GP tree representation and structural signatures
- gp_core.py: structure-based GP evolution, fitness, operators, run summaries
- requirements.txt: dependencies

## How to run with Docker 

### Build:

```bash
docker build -t cos710-a2 .
```

### Run (prev days):

```bash
docker run -v "${PWD}/data:/app/data" -v "${PWD}/output:/app/output" cos710-a2
```

### Run (prev vals): 

```bash
docker run -v "${PWD}/data:/app/data" -v "${PWD}/output:/app/output" cos710-a2 --data "data/Residential_Energy_Dataset_UK- 2014-2020.csv" --target-col Electricity_load --start-row 0 --max-rows 20000 --mode previous_values --lag-count 24 --runs 10 --P 120 --R 0.85,0.15 --S 4 --Dm 9 --DI 4 --Du 4 --Sg 30 --Dg 9 --Tg 0.75 --Wg 10 --DIg 3 --generations 60 --out-dir output_prev_values
```


## How To Run Locally 

### Requirements

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

- Params are included in the input for the terminal (feel free to adjust them before copy and pasting) 
  - please refer to 'Params used'
- Please ensure the dataset is in a folder called "data"

### 1. 10 runs

```bash
python gp_load_forecasting.py --data "data/Residential_Energy_Dataset_UK- 2014-2020.csv" --target-col Electricity_load --start-row 0 --max-rows 20000 --mode previous_days --lag-count 7 --points-per-day 0 --runs 10 --P 120 --R 0.85,0.15 --S 4 --Dm 9 --DI 4 --Du 4 --Sg 30 --Dg 9 --Tg 0.75 --Wg 10 --DIg 3 --generations 60 --out-dir output

```

### 2. Alternative - m previous values

```bash
python gp_load_forecasting.py --data "data/Residential_Energy_Dataset_UK- 2014-2020.csv" --target-col Electricity_load --start-row 0 --max-rows 20000 --mode previous_values --lag-count 24 --runs 10 --P 120 --R 0.85,0.15 --S 4 --Dm 9 --DI 4 --Du 4 --Sg 30 --Dg 9 --Tg 0.75 --Wg 10 --DIg 3 --generations 60 --out-dir output_prev_values
```
## Params used

- P: pop size
- R: crossover/ mutation rates
- S: tournament size
- Dm: max tree depth
- DI: initial tree depth
- Du: mutation max subtree depth
- Sg: number of generations for global search
- Dg: global area cut-off depth
- Tg: global similarity threshold index
- Wg: generations window of tolerance for no change
- DIg: initial tree depth in transferred global area


## Outputs Generated

- run_results.csv
- run_summary.csv

#### Columns in `run_results.csv`:

- run
- generation
- phase (global or local) 
- average_standardized_fitness
- average_tree_size
- variety_percentage
- average_hit_ratio

#### Columns in `run_summary.csv`:

- num_runs
- test_rmse_mean
- test_rmse_std
- test_rmse_best
- test_mae_mean
- test_mae_std
- test_mape_mean
- test_mape_std
- test_hit_ratio_mean
- test_hit_ratio_std
- test_hit_ratio_best
- hit_bound_used
- runtime_mean_seconds 
- runtime_std_seconds
- runtime_total_seconds
- best_run_index_1based
- best_expression

