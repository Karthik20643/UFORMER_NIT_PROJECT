# U-Former CBCT-to-Synthetic-CT Pipeline (Brain)

This project trains a 2D U-Former to convert brain CBCT scans into synthetic CT (sCT) scans. A single Python file, `uformer_pipeline.py`, handles the full workflow:

1. Preprocess 3D NIfTI volumes into 2D axial slices
2. Create a patient-level train, validation, and test split
3. Train the model
4. Stitch slice predictions back into 3D volumes and evaluate them
5. Generate sCT volumes for new CBCT scans

> **Note:** The model is 2D. It predicts each axial slice independently, and the slices are then stacked into a 3D volume.

## Method

| Item | Details |
|---|---|
| Input | CBCT slice |
| Target | Planning CT slice |
| HU window | Clipped to [-1000, 3000], then scaled to [0, 1] |
| Model input size | 256 × 256 |
| Output | Sigmoid output in [0, 1], converted using `HU = output × 4000 - 1000` |
| Loss | L1 loss inside the body mask |
| Metrics | MAE (HU), PSNR (dB), and SSIM |

## Installation

Install PyTorch for your system and CUDA version, then install the other dependencies:

```bash
pip install numpy SimpleITK scikit-image scikit-learn tqdm matplotlib
```

A GPU is strongly recommended. Mixed precision is used during training only when a GPU is available. Validation, testing, and inference run in full 32-bit precision.

## Data Layout

Place the data in this structure:

```text
Task2/
└── brain/
    └── <patient_id>/
        ├── cbct.nii.gz
        ├── ct.nii.gz
        └── mask.nii.gz
```

By default, only patient IDs starting with `2B` are used. The CBCT, CT, and mask for each patient must have the same shape. Patients with missing files or mismatched shapes are skipped.

## Quick Start

Run preprocessing, splitting, training, and testing:

```bash
python uformer_pipeline.py --phase all
```

## Running Individual Phases

```bash
# Preprocess the data
python uformer_pipeline.py --phase preprocess

# Create or check the patient split
python uformer_pipeline.py --phase split

# Train the model
python uformer_pipeline.py --phase train

# Train with custom settings
python uformer_pipeline.py --phase train --set NUM_EPOCHS=30 BATCH_SIZE=16

# Resume from the latest checkpoint
python uformer_pipeline.py --phase train --set RESUME=auto

# Test using best_model.pth
python uformer_pipeline.py --phase test

# Test a specific checkpoint
python uformer_pipeline.py --phase test --weights ./checkpoints/epoch_030.pth
```

## Inference

The input CBCT must match the training data format, including HU calibration and any resampling or cropping. The model only maps intensities. It doesn't perform registration or resampling.

```bash
# Single scan
python uformer_pipeline.py --phase infer --input cbct.nii.gz --output sct.nii.gz

# Folder of scans
python uformer_pipeline.py --phase infer --input ./cbct_folder --output ./sct_folder
```

The output keeps the size, spacing, origin, and direction of the input scan. Inference doesn't use a body mask, so the model's prediction is kept for the whole image.

## Running in a Notebook

```python
%run uformer_pipeline.py --phase train
```

You can also call it from Python:

```python
import uformer_pipeline as up

up.run(["train"], NUM_EPOCHS=30)
```

## Outputs

```text
Processed_Task2_Clean/
├── cbct/                     # Normalized CBCT slices
├── ct/                       # Normalized CT slices
├── mask/                     # Body mask slices
└── preprocess_meta.json      # HU window used for preprocessing

split.json                    # Patient-level data split

checkpoints/
├── best_model.pth            # Lowest validation MAE
├── last.pth                  # Latest epoch
├── epoch_XXX.pth             # Periodic checkpoints
├── training_log.csv
├── training_curves.png
└── config.json

Results/
├── test_sample.png
├── test_2d_metrics.json
├── test_3d_metrics.csv
├── test_summary.txt
└── NIfTI/
    ├── <patient_id>_sCT.nii.gz
    ├── <patient_id>_CT.nii.gz
    └── <patient_id>_CBCT.nii.gz
```

## Evaluation

All metrics are calculated in HU using the body mask.

- **2D slice metrics:** Calculated at the model size of 256 × 256 and averaged across non-empty slices.
- **3D patient metrics:** Calculated after predictions are resized to native size and stacked. Results are reported per patient and as mean ± standard deviation.
- **`SSIM`:** Sets voxels outside the mask to zero, then calculates SSIM across the whole volume. The matching background can make this score slightly optimistic.
- **`SSIM_in_mask`:** Calculates the SSIM map on the unmasked volumes, then averages it inside the mask.

## Configuration

Change settings in the `Config` class or override them from the command line:

```bash
python uformer_pipeline.py --phase train --set LR=0.0002 NUM_EPOCHS=100
```

| Setting | Default | Description |
|---|---|---|
| `RAW_DIR` | `./Task2/brain` | Raw patient data |
| `PROCESSED_DIR` | `./Processed_Task2_Clean` | Processed slices |
| `CHECKPOINT_DIR` | `./checkpoints` | Checkpoints and logs |
| `RESULTS_DIR` | `./Results` | Test outputs |
| `SPLIT_FILE` | `./split.json` | Saved patient split |
| `BRAIN_PREFIX` | `2B` | Patient ID prefix |
| `HU_MIN`, `HU_MAX` | `-1000`, `3000` | HU window |
| `IMG_SIZE` | `256` | Model input size |
| `SEED` | `42` | Random seed |
| `VAL_FRAC`, `TEST_FRAC` | `0.10`, `0.10` | Validation and test fractions |
| `TEST_IDS_OVERRIDE` | Empty | Comma-separated fixed test patient IDs |
| `EMBED_DIM` | `32` | Base feature size |
| `WIN_SIZE` | `8` | Attention window size |
| `BATCH_SIZE` | `8` | Training batch size |
| `LR` | `1e-4` | Learning rate |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay |
| `NUM_EPOCHS` | `60` | Total training epochs |
| `SAVE_EVERY` | `5` | Periodic checkpoint interval |
| `NUM_WORKERS` | `4` | Data loader workers |
| `AMP_TRAIN` | `True` | Mixed precision during training |
| `RESUME` | Empty | Empty, `auto`, or a checkpoint path |
| `EVAL_BATCH_SIZE` | `8` | Validation and 2D test batch size |
| `INFER_BATCH` | `16` | Slices per forward pass during 3D prediction |
| `WEIGHTS` | Empty | Checkpoint for testing or inference; defaults to `best_model.pth` |
| `SAVE_CBCT_NIFTI` | `True` | Save the input CBCT during testing |

`IMG_SIZE` must be divisible by `4 × WIN_SIZE`. The defaults meet this requirement.

## Important Notes

- **Changing preprocessing:** If you change the HU window, rerun preprocessing:

  ```bash
  python uformer_pipeline.py --phase preprocess --force
  ```

- **Keeping the split fixed:** `split.json` is reused on later runs so test patients stay separate from training.
- **Creating a new split:** Use `--phase split --force`, then retrain the model before testing it.
- **Using `--force` with `--phase all`:** This reprocesses all patients and creates a new split.
- **Resuming training:** `NUM_EPOCHS` is the total number of epochs, not the number of additional epochs.

## Limitations

- The model has no information from neighboring slices, which can cause inconsistencies in coronal and sagittal views.
- Slices are stacked without overlap, blending, or smoothing.
- Predictions are limited to the HU window of [-1000, 3000].
- Results depend on the input CBCT matching the training data format.

## Modifying the Code

| To change | Edit |
|---|---|
| Paths and hyperparameters | `Config` |
| Model architecture | `UFormer` and the blocks above it |
| Loss function | `build_criterion` |
| Optimizer or learning rate schedule | `phase_train` |
| Metrics | `slice_metrics` and `volume_metrics` |
| 3D prediction | `predict_volume01` |
