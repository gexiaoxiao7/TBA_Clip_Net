# TBA-ClipNet
This repository is the official implementation of the IJCNN2025 paper: 
"Using Domain Adapter and Prompt Learner for Few-shot Teaching Action Recognition in videos"

## Running Example

<table>
    <tr>
        <td ><center><img width="100%" alt="" src="https://github.com/gexiaoxiao7/MediaPool/blob/main/bowing_to_students.gif"/></center></td>
        <td ><center><img width="100%" alt="" src="https://github.com/gexiaoxiao7/MediaPool/blob/main/inviting_students.gif"/></center></td>
        <td ><center><img width="100%" alt="" src="https://github.com/gexiaoxiao7/MediaPool/blob/main/opearting.gif"/></center></td>
    </tr>
</table>

<table>
    <tr>
        <td ><center><img width="100%" alt="" src="https://github.com/gexiaoxiao7/MediaPool/blob/main/walking.gif"/></center></td>
        <td ><center><img width="100%" alt="" src="https://github.com/gexiaoxiao7/MediaPool/blob/main/pointting.gif"/></center></td>
        <td ><center><img width="100%" alt="" src="https://github.com/gexiaoxiao7/MediaPool/blob/main/writing.gif"/></center></td>
    </tr>
</table>

## Requirements
Create a conda environment and install dependencies:
```bash
conda create -n TBA-CLIPNet python=3.11
conda activate TBA-CLIPNet

# Install the according versions of torch and torchvision
pip install -r requirements.txt
```
Download pretrained yolov8 model file and put it under `Yolo-model` folder.

## Dataset
We suggest putting all datasets under the same folder (say $DATASET) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like
```text
$DATASET/
|–– tbad/
|–– ucf-101/
|–– humdb51/
```
Here is the link for our proposed teacher teaching behavior dataset(TBAD):
[TBAD DATASET](https://pan.baidu.com/s/1J1WaDKf_g42n-mDU4-XAOA?pwd=7g6n)
pwd: 7g6n
## Configs
The running configurations can be modified in `.yaml` files under `configs` folder path, including shot numbers, visual encoders, and hyperparamters. 
You should change the ROOT dir to your dataset dir.

## Running
For TBAD dataset:
```bash
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml
```

## Inferencing
For TBAD dataset:
```bash
python inference.py -cfg configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml -infer_set configs/inference.yaml
```
