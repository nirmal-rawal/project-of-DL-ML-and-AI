# Grocery Image Classification with CNN and Transfer Learning

This project implements Convolutional Neural Networks (CNNs) from scratch and utilizes transfer learning with pre-trained models (VGG16 and ResNet50) for classifying grocery items into three categories: Vegetables, Packages, and Fruit.

![Screenshot 2025-05-16 155042](https://github.com/user-attachments/assets/b492fe94-8713-4420-92ce-8b48b508c6c0)
![Screenshot 2025-05-16 155139](https://github.com/user-attachments/assets/ee9f0013-3644-4a4f-b84a-53f531bbf332)


## Dataset

The dataset contains images divided into three classes:
- Vegetables: 634 train, 85 validation, 587 test images
- Packages: 864 train, 100 validation, 781 test images
- Fruit: 1142 train, 111 validation, 1117 test images

## Implementation

### 1. Data Preparation
- Mounted Google Drive for data access
- Analyzed dataset distribution across classes
- Implemented extensive data augmentation:
  - Rotation (40° range)
  - Width/height shifting (30% range)
  - Shearing (30% range)
  - Zooming (30% range)
  - Horizontal/vertical flipping
  - Brightness adjustment (0.7-1.3x)

### 2. Model Architectures

#### Baseline CNN
- 3 Conv2D layers (32, 64, 128 filters)
- MaxPooling after each Conv layer
- 3 Dense layers (512, 256, 128 units)
- Achieved 83% test accuracy

#### Deeper CNN with Regularization
- 4 Conv blocks with increasing filters (32 → 256)
- Batch normalization after each Conv layer
- Dropout (increasing from 0.2 to 0.6)
- L2 regularization in dense layers
- Achieved 86% test accuracy

#### Transfer Learning Models
- **VGG16** (frozen weights) + custom head:
  - GlobalAveragePooling2D
  - Dense(512) with Dropout(0.5)
  - Achieved 90% test accuracy
- **ResNet50** (frozen weights) + custom head:
  - Similar architecture to VGG16
  - Achieved lower performance than VGG16

## Training Details

- **Optimizers**: Adam with learning rate scheduling
- **Batch Size**: 32
- **Image Size**: 224x224
- **Epochs**: 20-30 (with early stopping)
- **Metrics**: Accuracy and Categorical Crossentropy

## Results

| Model               | Validation Accuracy | Test Accuracy | Training Time |
|---------------------|---------------------|---------------|---------------|
| Baseline CNN        | 77.03%             | 83%           | ~53s/epoch    |
| Deeper CNN          | 81.08%             | 86%           | ~54s/epoch    |
| VGG16 (Transfer)    | 90.88%             | 90%           | ~74s/epoch    |
| ResNet50 (Transfer) | 53.04%             | -             | ~71s/epoch    |


