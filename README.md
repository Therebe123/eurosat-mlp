# eurosat-mlp-from-scratch

HW1: NumPy Three-Layer MLP for EuroSAT Classification

This repository implements a three-layer neural network classifier from scratch for EuroSAT RGB land-cover classification. It uses NumPy for matrix operations and manually implements forward propagation, backpropagation, SGD, learning-rate decay, cross-entropy loss, and L2 weight decay.

## Environment

Use the existing conda environment:

```powershell
conda run -n llm python --version
```

Expected Python dependencies are already installed in `llm`:

- `numpy`
- `Pillow`
- `matplotlib`

## Data

Place the EuroSAT dataset at:

```text
EuroSAT_RGB/
  AnnualCrop/
  Forest/
  HerbaceousVegetation/
  Highway/
  Industrial/
  Pasture/
  PermanentCrop/
  Residential/
  River/
  SeaLake/
```

Images are loaded at their original `64x64 RGB` size and flattened into `12288` input features. No resizing is applied.

## Train

Run the final training job used in the report:

```powershell
conda run -n llm python src/train.py --data_dir EuroSAT_RGB --epochs 20 --batch_size 128 --lr 0.01 --hidden_dim 256 --weight_decay 0.0001 --activation tanh --output_dir outputs
```

For a quick smoke test:

```powershell
conda run -n llm python src/train.py --data_dir EuroSAT_RGB --epochs 2 --hidden_dim 128 --activation relu --output_dir outputs_smoke --max_per_class 20
```

Training writes:

- `outputs/best_model.npz`
- `outputs/best_config.json`
- `outputs/history.json`

## Hyperparameter Search

Run the small grid search:

```powershell
conda run -n llm python src/search.py --data_dir EuroSAT_RGB --output_dir outputs --epochs 8
```

For a fast sanity check:

```powershell
conda run -n llm python src/search.py --data_dir EuroSAT_RGB --output_dir outputs_search_smoke --quick
```

The search writes `outputs/search_results.csv`.

## Test

Evaluate the best checkpoint:

```powershell
conda run -n llm python src/test.py --data_dir EuroSAT_RGB --checkpoint outputs/best_model.npz --output_dir outputs
```

Testing writes:

- `outputs/confusion_matrix.png`
- `outputs/test_predictions.json`

## Visualize

Generate report figures:

```powershell
conda run -n llm python src/visualize.py --data_dir EuroSAT_RGB --output_dir outputs
```

Visualization writes:

- `outputs/loss_curve.png`
- `outputs/accuracy_curve.png`
- `outputs/first_layer_weights.png`
- `outputs/error_examples.png`

## Report and Weights

- Final validation accuracy: `0.5081`
- Final test accuracy: `0.5156`
- GitHub repository: <https://github.com/Therebe123/eurosat-mlp>
- Model weights: <https://drive.google.com/file/d/1VuDTOUvUjob4U2cKUk6uWAT-6IMhckJv/view?usp=drive_link>

The experiment report is submitted separately and is intentionally not tracked in this repository.
