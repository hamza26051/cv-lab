# Computer Vision Lab 08 - Object Detection with Faster R-CNN

An **already trained and tested** PyTorch/TorchVision Faster R-CNN project for Lab 08, task 6.1. The original manual prohibits using BCCD. This project instead uses an **original, reproducible synthetic dataset** of colored geometric signs with three object classes: `circle`, `square`, `triangle`. Nothing needs to be downloaded to create the dataset.

> **Important academic scope:** This is a controlled synthetic dataset with regular geometry and simple backgrounds, **not a real traffic-sign dataset**. Its high mAP should not be presented as evidence of real-world generalization. The project trains an actual RPN/RoI-based Faster R-CNN detector, rather than a hand-coded color classifier.

## What is in the project?

- `Lab08_Faster_RCNN.ipynb` - step-by-step runnable Jupyter notebook with explanation, generation, training, evaluation, figures and inference.
- `lab08.py` - complete source: dataset generation, bounding-box annotations, custom PyTorch Dataset, Faster R-CNN model, training, AP/mAP evaluation, visualization and inference.
- `data/train`, `data/val`, `data/test` - already generated images (150 / 30 / 30).
- `data/train.csv`, `data/val.csv`, `data/test.csv` - annotations in `filename,class,xmin,ymin,xmax,ymax` format.
- `outputs/best_model.pth` - trained detector checkpoint.
- `outputs/training_history.csv`, `training_history.json`, `training_summary.json` - recorded training run.
- `outputs/test_metrics.json`, `test_detections.csv` - recorded held-out results and boxes.
- `figures/` - class-count chart, labeled training images, training curves, predictions on held-out images.
- `Lab08_Faster_RCNN_Report.pdf` - submission-ready report; put your name and roll number in the blanks.
- `make_report.py` - reproducibly rebuilds the PDF from saved experiment results.
- `run_windows.bat`, `run_linux_mac.sh` - launchers that install dependencies then run the complete pipeline.

## Existing measured results

The exact saved checkpoint was trained on CPU using seed 42, 12 epochs, Adam, batch size 4, and learning rate 0.001. The 30-image test split has 55 annotated objects.

| Metric | Test set |
|---|---:|
| mAP@IoU 0.50 | 1.0000 |
| mAP@IoU 0.75 | 1.0000 |
| Precision @ score threshold 0.35 | 1.0000 |
| Recall @ score threshold 0.35 | 1.0000 |
| F1 @ score threshold 0.35 | 1.0000 |
| Mean matched IoU | 0.9122 |
| True positives / false positives / false negatives | 55 / 0 / 0 |

The detector reports `background=0` and the 3 foreground classes `circle=1`, `square=2`, `triangle=3`. All metrics are from **actual executed Python evaluation**, not estimated values.

## Run on Windows PowerShell

```powershell
cd "path\to\Lab08_RCNN_Object_Detection"
py -m pip install -r requirements.txt
py lab08.py all --n-train 150 --n-val 30 --n-test 30 --seed 42 --epochs 12
```

Or double-click `run_windows.bat` (requires Python installed and available on PATH).

## Run on Linux or macOS

```bash
cd Lab08_RCNN_Object_Detection
python3 -m pip install -r requirements.txt
python3 lab08.py all --n-train 150 --n-val 30 --n-test 30 --seed 42 --epochs 12
```

## Run *without* retraining

The project already includes the generated data and checkpoint, so to re-check the test set:

```bash
python lab08.py test
```

To predict on a specific image (use a generated image first):

```bash
python lab08.py predict --image data/test/test_0001.png
```

The inference output is saved as `outputs/test_0001_prediction.png` and `outputs/test_0001_prediction.json`.

## Open the notebook

```bash
python -m pip install notebook
jupyter notebook Lab08_Faster_RCNN.ipynb
```

It regenerates the dataset, trains, and evaluates when its cells are run from top to bottom. **The provided notebook is runnable but has not been saved with executed cell outputs**; the code, model and output files were instead executed and verified using the CLI. Avoid rerunning generation alone with a different seed and then treating saved metrics as belonging to that new dataset.

## CSV format

Every object is represented as one row, so a picture with three shapes has three annotations:

```
filename,class,xmin,ymin,xmax,ymax
train_0000.png,square,52,27,84,59
train_0000.png,circle,41,80,76,115
```

These are transformed into PyTorch `boxes` and `labels` tensors. Coordinates are pixel-based `xyxy`; they are always valid boxes within the 128 x 128 image.

## Model details

- Shared trainable CNN backbone (4 convolutional layers, 64 output channels).
- Region Proposal Network (RPN) with anchors of sizes 16, 24, 32, 48 and 64 pixels.
- MultiScaleRoIAlign, a 128-unit two-layer classifier/regressor head.
- 4 total class indices (background and 3 foreground classes).
- Four detection losses: objectness, RPN regression, ROI classification, ROI regression.
- Image-level train/validation/test separation and horizontal flip augmentation only on training data.
- AP evaluated by greedy one-to-one class-aware IoU matching, and precision-envelope AP integration.

For larger natural-image datasets, use a more capable backbone and realistic annotation / augmentation. Synthetic 128-pixel images don't show production-ready performance.

## Source notes

The course manual provides the Faster R-CNN workflow and requires a different dataset for task 6.1. The PyTorch custom dataset interface follows the detection tutorial at <https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html>. The detection model uses the TorchVision FasterRCNN implementation.
