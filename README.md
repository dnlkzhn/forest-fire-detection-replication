# Replication of "A Comparative Assessment of CNN-Sigmoid and CNN-SVM Model for Forest Fire Detection"

## Replication Overview

This project replicates the paper titled **"A Comparative Assessment of CNN-Sigmoid and CNN-SVM Model for Forest Fire Detection"**.  
The main goal is to validate the performance of two different deep learning models for forest fire detection:

- **CNN-Sigmoid**: A traditional convolutional neural network with a sigmoid output activation.
- **CNN-SVM**: A hybrid approach where a CNN is used as a feature extractor and a Support Vector Machine (SVM) serves as the classifier.

The original study focuses on early forest fire detection using drone-captured images, aiming to replace conventional methods that are labor-intensive and slow.

### Flowchart of the Original Approach
The original CNN-SVM model flow:
1. Input Image → CNN Feature Extraction
2. Flattened Features → SVM Classification
3. Output: Fire / No-Fire Prediction

---

## Implementation Details

The replication was carried out using:
- **Dataset**: Forest Fire Dataset with 1898 images (fire / no-fire).
- **Frameworks**: TensorFlow (for CNN), scikit-learn (for SVM).
- **Environment**: Google Colaboratory (Colab) with GPU acceleration.

Two models were implemented:

- **CNN-Sigmoid**: 
  - Standard CNN with convolutional layers, max-pooling, dropout, and a fully connected sigmoid output layer.
  - Optimized with Adam optimizer and binary cross-entropy loss.
  
- **CNN-SVM**:
  - Used the CNN feature extractor from the CNN-Sigmoid model.
  - Removed the output layer and extracted flattened features.
  - Trained a separate linear SVM classifier on these features using hinge loss.

Both models were trained for **20 epochs** with a **batch size of 32**.  
During the implementation, additional validation splits and standardized feature scaling were introduced to stabilize training and improve evaluation accuracy.

---

## Evaluation and Results

The replication successfully validated the original paper’s findings, with both models achieving high classification performance.  
**Key results:**
- **CNN-Sigmoid Test Accuracy**: 95.45%
- **CNN-SVM Test Accuracy**: 95.26%
- **Precision, Recall, and F1-Score**: Around 0.95 for both models.

Despite matching the general trends, several issues in the original article were uncovered:
- The article incorrectly claims that sigmoid outputs can be less than zero.
- The confusion matrix interpretation contained labeling mistakes.
- The authors proposed training SVMs in epochs, which is unconventional for SVMs.
- Models were trained for an excessive number of epochs, leading to overfitting.
- Data augmentation methods were vaguely described and difficult to replicate exactly.

**Conclusion:**  
While the CNN-SVM model slightly underperformed compared to the original results, the replication confirmed that using an SVM as a classifier improves fire detection robustness. Several critical mistakes in the original paper were identified, and suggestions for further improvements, such as greyscale preprocessing, trying non-linear SVM kernels, and better augmentation, were proposed.

---
