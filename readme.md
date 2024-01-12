# Salient Text Detection

## Dataset Preparation

- Download original e-commerce dataset [here](https://github.com/leafy-lee/E-commercial-dataset)

- Download improved text annotations [here](https://drive.google.com/drive/folders/1hjCa3f85c3zDcoQYiMx-nr9nLV3Qa36a?usp=drive_link)

- Pretrained weights available [here](https://drive.google.com/file/d/1VMMnKXaC9FLycjzHfVnE1L--dd0pOR47/view?usp=drive_link)

## Installation
```
python = 3.9.0
pytorch = 1.10.0+cu111
detectron2 = 0.6+cu113
```
## Train the model

To train saliency and text model:

```
python main.py --config configs/ecom.yaml
```

To train text only model:

```
python main_text.py --config configs/ecom_text.yaml
```

