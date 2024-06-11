# Real-and-Fake-İmage-Detection


This repository contains code for training and evaluating a combined deep learning model that utilizes ShuffleNet and SqueezeNet for binary image classification. The model is trained to differentiate between real and fake images.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Data Augmentation and Preprocessing](#data-augmentation-and-preprocessing)
- [Training, Validation, and Testing](#training-validation-and-testing)
- [Visualization](#visualization)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
## Model Architecture

The combined model leverages the strengths of ShuffleNet and SqueezeNet. The architecture is as follows:

- **ShuffleNet**: The final fully connected layer is removed and replaced with an identity layer.
- **SqueezeNet**: The classifier's final convolutional layer is modified to output a single feature map.

The outputs from both networks are concatenated and passed through a new fully connected layer to produce the final binary classification.

```python
class CombinedModel(nn.Module):
    def __init__(self, shufflenet, squeezenet):
        super(CombinedModel, self).__init__()
        self.shufflenet = shufflenet
        self.squeezenet = squeezenet
        self.fc = nn.Linear(num_ftrs_shufflenet + 1, 1)
        
    def forward(self, x):
        shufflenet_output = self.shufflenet(x)
        squeezenet_output = self.squeezenet(x)
        squeezenet_output = squeezenet_output.view(squeezenet_output.size(0), -1)
        combined_output = torch.cat((shufflenet_output, squeezenet_output), dim=1)
        return self.fc(combined_output)
```

## Data Augmentation and Preprocessing

The dataset is loaded using `torchvision.datasets.ImageFolder`. Several data augmentation techniques are applied to increase the robustness of the model:

- Random resized cropping
- Horizontal and vertical flipping
- Rotation
- Color jitter
- Gaussian blur
- Random grayscale conversion
- Normalization

```python
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Training, Validation, and Testing

The dataset is split into training (80%), validation (10%), and testing (10%) sets. The model is trained using binary cross-entropy loss and the Adam optimizer. The training process includes:

- Forward pass
- Loss calculation
- Backward pass and optimization
- Accuracy calculation

Validation and testing follow a similar process but without backpropagation.

## Visualization

Training, validation, and test metrics are visualized using `matplotlib`. Accuracy and loss are plotted over the number of epochs to monitor the model's performance.

```python
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
```

## Requirements

- Python 3.7+
- PyTorch
- Torchvision
- Matplotlib
- PIL

Install the required packages using:

```bash
pip install torch torchvision matplotlib pillow
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/aakcay5656/Real-and-Fake-İmage-Detection.git
cd Real-and-Fake-face-detection
```

2. Organize your dataset in the following structure:

```
dataset/
└── data5_h/
    └── train/
        ├── class1/
        └── class2/
```



## Results

The training and validation accuracy and loss metrics will be printed at each epoch and saved as plots in the output directory. The model will also save annotated images with predictions and true labels.



## Acknowledgements

- The ShuffleNet and SqueezeNet models are provided by the PyTorch model library.
- Data augmentation techniques are implemented using `torchvision.transforms`.

## License

This project is licensed under the MIT License.

