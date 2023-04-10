
# Lung cancer detection

This project aims to detect lung cancer from CT-Scan images using deep learning techniques. The dataset used in this project contains CT-Scan images of Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, and normal cells. The dataset can be found [here](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images?resource=download).

## Dataset Description

The CT-Scan images are in jpg or png format to fit the model. The dataset contains four main folders:

-   `Adenocarcinoma`: contains CT-Scan images of Adenocarcinoma of the lung. Adenocarcinoma is the most common form of lung cancer, accounting for 30% of all cases overall and about 40% of all non-small cell lung cancer occurrences.
    
-   `Large cell carcinoma`: contains CT-Scan images of Large-cell undifferentiated carcinoma of the lung. This type of lung cancer usually accounts for 10 to 15% of all cases of NSCLC.
    
-   `Squamous cell carcinoma`: contains CT-Scan images of Squamous cell carcinoma of the lung. This type of lung cancer is responsible for about 30% of all non-small cell lung cancers, and is generally linked to smoking.
    
-   `Normal`: contains CT-Scan images of normal cells.
    

The dataset is divided into three sets: training, testing, and validation. The `training` set contains 70% of the data, the `testing` set contains 20% of the data, and the `validation` set contains 10% of the data.

## Technologies Used

This project was implemented using the following technologies:

-   `TensorFlow` and `Keras`: for building and training the deep learning model.
-   `ImageDataGenerator`: for data augmentation.
-   `ResNet50`, `VGG16`, `ResNet101`, `VGG19`, `DenseNet201`, `EfficientNetB4`, `MobileNetV2`: pre-trained models used for transfer learning.
-   `PIL`, `OpenCV`: for image processing.
-   `Matplotlib`: for visualizing the training and validation results.

## Model Architecture

The deep learning model used in this project is a convolutional neural network (CNN). The model consists of several layers including convolutional, max pooling, batch normalization, and dense layers. Transfer learning is used by initializing the model with pre-trained weights from one of the above-mentioned pre-trained models.

The model was trained using the `Adam` optimizer with a learning rate of 0.001, a batch size of 16, and a total of 50 epochs.

## Results

The model achieved an accuracy of 95.2% on the testing set. The training and validation accuracies and losses for each epoch are visualized below:

![Accuracy and Loss](https://raw.githubusercontent.com/BarriBarri20/Lung-cancer-detection-model-training/main/accuracy_epochs.png)

## Checkpoints

Two checkpoints were saved during the training process. These checkpoints can be used to resume training or to evaluate the model on new data.

## Credit

This notebook was created by the Lovelace team,, for the AI_NIGHT_CHALLENGE competition.
