# Prompted Segmentation for Drywall QA
Given an image of a crack region or wall joint with a text prompt, the model will generate binary mask of that area. 

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PushpakAg/Text-Conditioned-Drywall-QA.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Data Preparation
Using the provided link download the dataset in Pascal VOC format and put in under `data/raw`. 

- [Taping Area](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect)
- [Cracks](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36)


## Usage

1. **Prepare the data:**
   ```bash
   python prepare_data.py
   ```
2. **Train the model:**
- Check `config.py`
   ```bash
   python train.py
   ```
3. **Make predictions:**
   The `predict.py` script can be used in two modes: `predict` for single image inference and `eval` for evaluating the model on the test set.

   - **Prediction Mode:**
     Use `--mode predict` along with `--image` (path to the input image) and `--prompt` (text prompt for segmentation).
     ```bash
     python predict.py --mode predict --image images/examples/crack_001.jpg --prompt "segment crack"
     ```

   - **Evaluation Mode:**
     Use `--mode eval` to run evaluation on the test dataset.
     ```bash
     python predict.py --mode eval
     ```
   Optionally, you can specify a different model checkpoint using `--checkpoint`.
   ```bash
   python predict.py --mode predict --image images/crack.jpg --prompt "segment crack" --checkpoint outputs/checkpoints/best_model.pth.tar
   ```

## Results
- ![Crack](images/crack.jpg)
- ![Taping Area](images/taping_area.jpg)