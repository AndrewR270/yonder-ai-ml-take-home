# Methodology

This is a **detection** pipeline for mallets and bottles, using Roboflow datasets and YOLO v8 models for image analysis.

My best.pt tensor was found after *40 epochs, 2.990 hours*, with a final mean Average Precision **mAP50 of 0.9115**. This means 91.15%of the time, my ML model satisfies the IoU (Intersection over Union) threshold of 0.50 - the bounding boxes sufficiently overlap with the ground truth. Specifically, **mallets** had a mAP50 of **0.967**, while **bottles** had a mAP50 of **0.856**.

This training took place on an Intel Core 5 120U (1.40 GHz) CPU with 10 Cores, with (roughly) *~37.6 Wh of power used*. Inference latency averaged **68.6 ms/image or ~14.5 FPS**.

## 1. How to run it

### Setup

Clone this repo:

```bash
git clone https://github.com/AndrewR270/yonder-ai-ml-take-home.git
cd yonder-ai-ml-take-home
```

Install **Conda**: ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)) and create a conda environment:

```bash
conda create -n yonder python=3.12 -y
conda activate yonder
```

Install dependencies: 

```bash
pip install -r requirements.txt
```

All my versions and packages such as  `ultralytics roboflow opencv-python matplotlib python-dotenv truststore` have already been saved to requirements.txt using `pip freeze > requirements.txt`. 

### Dataset

Got to Roboflow and get the Sampled YD Object Detection Computer Vision Dataset as a code snippet. Create an /.env and add your Roboflow API key:

```bash
ROBOFLOW_API_KEY=""
```

To get the training data, run **src/get_dataset.py**: 

```bash
python src/get_dataset.py
```

### Training

To train the model, run **src/train.py**:

```bash
python src/train.py
```

Run with `-h` or `--help` to see all options for the training script.

By default, `epochs=40`, `imgsz=512`, and `batch=16`.


### Video Frame Extraction (Optional)

This is a helper script if you want to extract frames from a video:

```bash
python utils/extract_frames.py --video /path/to/video
```

Run with `-h` or `--help` to see all options for the training script.

You must supply a path to a video.

By default, `output` resolves to `frames/last`, and `stride=10` (every 10th frame).

### Inference

To run object detection on a folder of test images and generate YOLO label files, run **src/inference.py**:

```bash
python src/inference.py --input /path/to/image/folder
```

Run with `-h` or `--help` to see all options for the inference script.

You must supply paths to an input image folder.

By default, `output` resolves to `/inferences/last` and `weights` resolves to `weights/best.pt`.

## 2. Thought process

### Pipeline

Mirroring past PyTorch ML projects I have done, the three main functions (represented by the three src/ scripts) are: data acquisition, processing and training, and prediction. While I have separated processing and training in the past, YOLO can handle image augmentation and resizing natively, so only one script was used for both.

Overall the project follows this trajectory:

1. Environment Setup (Conda / Pip Dependencies)
2. Dataset Fetching (Roboflow API & Key Injection)
3. Model Training & Validation (YOLOv8 Fine-Tuning)
4. Performance Metrics Logging & Diagnostic Output Generation
5. Real-World Field Video Test & Frame Inference
6. Standalone Inference Execution (inference.py -> .txt Label Generation)

### Image Augmentation

Beyond just image resizing, I included the following image augmentations on the YOLO model to better represent the different lighting and orientation conditions that the rover and its camera may encounter.

- **Hue** (*hsv_h=0.015*): I wanted to emphasize the edge and shape detection capabilities of the YOLO model in training and thus changed the hue channels by a random value between -0.015 and +0.015 to make training less reliant on specific color during repeated passes.

- **Saturation** (*hsv_s=0.7*): Color saturation may change when lighting conditions differ, being less vibrant at lower light levels. Dust and clouding on the camera lens in Martian storms also affects the color which the lenses percieve. To allow each image in the dataset to reflect this I applied saturation changes between -70% and +70%.

- **Brightness** (*hsv_v=0.4*): Since lighting conditions can change on Earth and Mars, with shadows and times of day producing different brightnesses for cameras, I implemented a -40% and +40% random brightness change. I did not want to go beyond this as, from my research, ground imaging is not common at night. While modern Mars rovers can still operate at night, they spend more energy keeping themselves warm and prioritize sky imaging instead.

- **Rotation** (*degrees=15.0*): A Mars rover is very commonly going to be moving over rocky, dusty, uneven terrain where it may encounter objects and features at different angles, so a random -15.0 to 15.0 degree augmentation was added.

- **Mirroring** (*fliplr=0.5*): Added for extra variance, 50% chance.

## 3. Known limitations

These are ranked in terms of which I consider to be most pressing.

### 1) Lack of Scale Augmentation

I did not apply **scale** as an augmenting factor in the training script. This was due to an oversight on my part. Adding scale would allow the same image to be used to represent different distances across training epochs, reflecting how the rover would recognize an object at different distances.

I refrained from retraining the tensors due to time constraints, but this is a high-priority addition, as it allows us as ML engineers to track confidence intervals and bounding boxes as our "camera" operates when the rover approaches the object.

### 2) Disparities in Object Type Recognition

The difference between **mallet** and **bottle** recognition are notable, with *bottles having a 0.111 lower mAP50 score than mallets*. I suspect this is due to the much larger volume of mallets available in training, and an immediate solution to this might be to add more bottle images.

However, as the README notes, the size of the dataset does not mean it is better. Bottles are more sensitive to light changes and are harder to pick out in an environment, so training more with added scale augmentations can help with this.

Additionally, as images were only 512x512, upscaling by providing a larger *imgsz* can also help to pick out finer details.

### 3) NPU Efficiency

On my CPU, inference latency was around 14.5 FPS and used 37.6Wh. This is slower than optimal for the rover, which would have less power to perform these computations and the FPS would only drop further. This was bottlenecked because I had no CUDA acceleration and only used x86 processor cores.

Running the full PyTorch model would not be time and energy effieicnt. My research has said that quantizing the best weights to INT8/FP16 would help improve efficiency, with extensive improvement possible by converting to *.rknn* binary files, as OrangePi NPUs run with Rockchip processors compatible with Rockchip Neural Network.

### 4) Training Time

As I used my own CPU, training was slower and more inefficient, making repeated fine-tuning training sessions more costly. My CPU used 0 workers and required nearly 3.0 hours for 40 epochs. Leveraging CUDA GPU acceleration would drop training time down to minutes, enabling deeper training.

### 5) Customizability

Adding a CLI arg for training scripts to add custom names to output folders would allow us to name training result folders for objects other than mallets and bottles.
