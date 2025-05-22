
# Brain Tumor Detection Using Transfer Learning

This project uses a ResNet50-based deep learning model to classify brain tumors into four categories: glioma, meningioma, pituitary, and no tumor. Images are preprocessed and augmented using Keras’ ImageDataGenerator to improve model robustness. The model is fine-tuned on medical images with dropout layers to prevent overfitting.

Training was done for 25 epochs with the Adam optimizer, achieving **94.25% training accuracy** and **86.88% testing accuracy**. The model performs well in precision, recall, and F1-score, especially for the 'no tumor' and 'pituitary' classes.

A REST API enables real-time predictions by accepting input images and returning tumor class probabilities. The application is containerized using Docker and deployed on [Hugging Face Spaces](https://huggingface.co/spaces/Shravinya/Brain_Tumor_Detection) with GPU support for fast and accessible inference.

**Limitations:** The model may underperform on low-resolution or noisy images and requires more diverse data for clinical deployment.

---

## Features
- Transfer learning with ResNet50 pretrained on ImageNet  
- Data augmentation to prevent overfitting  
- REST API for inference  
- Docker containerization  
- GPU-enabled deployment on Hugging Face Spaces

