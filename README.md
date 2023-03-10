<p align="center"><img width="100%" src="ML/others/logo/torch_and_tf.svg" /></p>

--------------------------------------------------------------------------------


[![Build Status](https://travis-ci.com/aladdinpersson/Machine-Learning-Collection.svg?branch=master)](https://travis-ci.com/avs-abhishek123/Machine-Learning) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[logo]: https://github.com/avs-abhishek123/Machine-Learning/blob/main/ML/others/logo/youtube_logo.png

# Machine Learning Collection
In this repository you will find tutorials and projects related to Machine Learning. I try to make the code as clear as possible, and the goal is be to used as a learning resource and a way to lookup problems to solve specific problems. If you got any questions or suggestions for future videos I prefer if you ask it on [YouTube](https://www.youtube.com/channel/UCFf3DOwYUF2Rhe2-rC3uy9A). This repository is contribution friendly, so if you feel you want to add something then I'd happily merge a PR :smiley:

## Table Of Contents
- [Machine Learning Algorithms](#machine-learning)
- [PyTorch Tutorials](#pytorch-tutorials)
	- [Basics](#basics)
	- [More Advanced](#more-advanced)
    - [Object Detection](#object-detection)
	- [Generative Adversarial Networks](#generative-adversarial-networks)
	- [Architectures](#architectures)
- [TensorFlow Tutorials](#tensorflow-tutorials)
	- [Beginner Tutorials](#beginner-tutorials)
	- [CNN Architectures](#cnn-architectures)

## Machine Learning
* [![Youtube Link][logo]]() &nbsp; [Linear Regression](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/algorithms/LinearRegression/linear_regression_gradient_descent.py) **- With Gradient Descent** :white_check_mark: 
* [![Youtube Link][logo]]() &nbsp; [Linear Regression](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/algorithms/LinearRegression/linear_regression_normal_equation.py) **- With Normal Equation** :white_check_mark:
* [![Youtube Link][logo]]() &nbsp; [Logistic Regression](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/algorithms/LogisticRegression/logistic_regression.py)
* [![Youtube Link][logo]]() &nbsp; [Naive Bayes](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/algorithms/NaiveBayes/naivebayes.py) **- Gaussian Naive Bayes**
* [![Youtube Link][logo]]() &nbsp; [K-nearest neighbors](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/algorithms/KNN/knn.py)
* [![Youtube Link][logo]]() &nbsp; [K-means clustering](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/algorithms/KMeans/kmeansclustering.py) 
* [![Youtube Link][logo]]() &nbsp; [Support Vector Machine](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/algorithms/SVM/svm.py) **- Using CVXOPT**
* [![Youtube Link][logo]]() &nbsp; [Neural Network](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/algorithms/NeuralNetwork/NN.py)
* [Decision Tree](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/algorithms/DecisionTree/decision_tree.py)

## PyTorch Tutorials
If you have any specific video suggestion please make a comment on YouTube :)

### Basics
* [![Youtube Link][logo]]() &nbsp; [Tensor Basics](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_tensorbasics.py)
* [![Youtube Link][logo]]() &nbsp; [Feedforward Neural Network](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_simple_fullynet.py)
* [![Youtube Link][logo]]() &nbsp; [Convolutional Neural Network](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_simple_CNN.py)
* [![Youtube Link][logo]]() &nbsp; [Recurrent Neural Network](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py)
* [![Youtube Link][logo]]() &nbsp; [Bidirectional Recurrent Neural Network](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_bidirectional_lstm.py)
* [![Youtube Link][logo]]() &nbsp; [Loading and saving model](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_loadsave.py)
* [![Youtube Link][logo]]() &nbsp; [Custom Dataset (Images)](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/custom_dataset)
* [![Youtube Link][logo]]() &nbsp; [Custom Dataset (Text)](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/custom_dataset_txt)
* [![Youtube Link][logo]](h) &nbsp; [Mixed Precision Training](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_mixed_precision_example.py)
* [![Youtube Link][logo]]() &nbsp; [Imbalanced dataset](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/Imbalanced_classes)
* [![Youtube Link][logo]]() &nbsp; [Transfer Learning and finetuning](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_pretrain_finetune.py)
* [![Youtube Link][logo]]() &nbsp; [Data augmentation using Torchvision](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_transforms.py)
* [![Youtube Link][logo]]() &nbsp; [Data augmentation using Albumentations](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/albumentations_tutorial)
* [![Youtube Link][logo]]() &nbsp; [TensorBoard Example](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_tensorboard_.py)
* [![Youtube Link][logo]]() &nbsp; [Calculate Mean and STD of Images](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_std_mean.py)
* [![Youtube Link][logo]]() &nbsp; [Simple Progress bar](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_progress_bar.py)
* [![Youtube Link][logo]]() &nbsp; [Deterministic Behavior](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/set_deterministic_behavior/pytorch_set_seeds.py)
* [![Youtube Link][logo]]() &nbsp; [Learning Rate Scheduler](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_lr_ratescheduler.py) 
* [![Youtube Link][logo]]() &nbsp; [Initialization of weights](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/Basics/pytorch_init_weights.py)

### More Advanced
* [![Youtube Link][logo]]() &nbsp; [Text Generating LSTM](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Projects/text_generation_babynames/generating_names.py)
* [![Youtube Link][logo]]() &nbsp; [Semantic Segmentation w. U-NET](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet)
* [![Youtube Link][logo]]() &nbsp; [Image Captioning](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/more_advanced/image_captioning)
* [![Youtube Link][logo]]() &nbsp; [Neural Style Transfer](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/more_advanced/neuralstyle/nst.py)
* [![Youtube Link][logo]]() &nbsp; [Torchtext [1]](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial1.py) [Torchtext [2]](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial2.py) [Torchtext [3]](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial3.py)
* [![Youtube Link][logo]]() &nbsp; [Seq2Seq](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/more_advanced/Seq2Seq/seq2seq.py) **- Sequence to Sequence (LSTM)**
* [![Youtube Link][logo]]() &nbsp; [Seq2Seq + Attention](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/more_advanced/Seq2Seq_attention/seq2seq_attention.py) **- Sequence to Sequence with Attention (LSTM)**
* [![Youtube Link][logo]]() &nbsp; [Seq2Seq Transformers](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/more_advanced/seq2seq_transformer/seq2seq_transformer.py) **- Sequence to Sequence with Transformers**
* [![Youtube Link][logo]]() &nbsp; [Transformers from scratch](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py) **- Attention Is All You Need**

### Object Detection
[Object Detection Playlist]()
* [![Youtube Link][logo]]() &nbsp; [Intersection over Union](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/object_detection/metrics/iou.py) 
* [![Youtube Link][logo]]() &nbsp; [Non-Max Suppression](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/object_detection/metrics/nms.py)
* [![Youtube Link][logo]]() &nbsp; [Mean Average Precision](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/object_detection/metrics/mean_avg_precision.py)
* [![Youtube Link][logo]]() &nbsp; [YOLOv1 from scratch](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/object_detection/YOLO)
* [![Youtube Link][logo]]() &nbsp; [YOLOv3 from scratch](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3)

### Generative Adversarial Networks
[GAN Playlist](https://youtube.com/playlist?list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va)

* [![Youtube Link][logo]]() &nbsp; [Simple FC GAN](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py)
* [![Youtube Link][logo]]() &nbsp; [DCGAN](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/2.%20DCGAN)
* [![Youtube Link][logo]]() &nbsp; [WGAN](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/3.%20WGAN)
* [![Youtube Link][logo]]() &nbsp; [WGAN-GP](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/4.%20WGAN-GP)
* [![Youtube Link][logo]]() &nbsp; [Pix2Pix](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/Pix2Pix)
* [![Youtube Link][logo]]() &nbsp; [CycleGAN](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/CycleGAN)
* [![Youtube Link][logo]]() &nbsp; [ProGAN](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/ProGAN)
* [SRGAN](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/SRGAN)
* [ESRGAN](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/ESRGAN)
* [StyleGAN](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/StyleGAN) - NOTE: NOT DONE



### Architectures
* [![Youtube Link][logo]]() &nbsp; [LeNet5](https://github.com/avs-abhishek123/Machine-Learning/blob/2b0a5061275151fdd308db8a65a0193c55ee42ab/ML/Pytorch/CNN_architectures/lenet5_pytorch.py#L12-L52) **- CNN architecture**
* [![Youtube Link][logo]]() &nbsp; [VGG](https://github.com/avs-abhishek123/Machine-Learning/blob/21a44a7aac0e732d0c21f520f265297fc8abc13e/ML/Pytorch/CNN_architectures/pytorch_vgg_implementation.py#L12-L106) **- CNN architecture**
* [![Youtube Link][logo]]() &nbsp; [Inception v1](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/CNN_architectures/pytorch_inceptionet.py) **- CNN architecture**
* [![Youtube Link][logo]]() &nbsp; [ResNet](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/CNN_architectures/pytorch_resnet.py) **- CNN architecture**
* [![Youtube Link][logo]]() &nbsp; [EfficientNet](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/Pytorch/CNN_architectures/pytorch_efficientnet.py) **- CNN architecture**

## TensorFlow Tutorials
If you have any specific video suggestion please make a comment on YouTube :)

### Beginner Tutorials
* [![Youtube Link][logo]]() &nbsp; Tutorial 1 - Installation, Video Only
* [![Youtube Link][logo]]() &nbsp; [Tutorial 2 - Tensor Basics](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial2-tensorbasics.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 3 - Neural Network](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial3-neuralnetwork.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 4 - Convolutional Neural Network](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial4-convnet.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 5 - Regularization](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial5-regularization.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 6 - RNN, GRU, LSTM](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial6-rnn-gru-lstm.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 7 - Functional API](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial7-indepth-functional.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 8 - Keras Subclassing](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial8_keras_subclassing.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 9 - Custom Layers](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial9-custom-layers.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 10 - Saving and Loading Models](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial10-save-model.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 11 - Transfer Learning](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial11-transfer-learning.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 12 - TensorFlow Datasets](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial12-tensorflowdatasets.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 13 - Data Augmentation](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial13-data-augmentation.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 14 - Callbacks](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial14-callbacks.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 15 - Custom model.fit](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial15-customizing-modelfit.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 16 - Custom Loops](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial16-customloops.py)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 17 - TensorBoard](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial17-tensorboard)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 18 - Custom Dataset Images](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial18-customdata-images)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 19 - Custom Dataset Text](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial19-customdata-text)
* [![Youtube Link][logo]]() &nbsp; [Tutorial 20 - Classifying Skin Cancer](https://github.com/avs-abhishek123/Machine-Learning/tree/main/ML/TensorFlow/Basics/tutorial20-classify-cancer-beginner-project-example) **- Beginner Project Example**

### CNN Architectures
* [LeNet](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/CNN_architectures/LeNet5)
* [AlexNet](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/CNN_architectures/AlexNet)
* [VGG](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/CNN_architectures/VGGNet)
* [GoogLeNet](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/CNN_architectures/GoogLeNet)
* [ResNet](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/CNN_architectures/ResNet)
