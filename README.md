# Human Feedback Based Image Captioning

## Table of Contents
- [Documentation](#documentation)
- [Description](#description)
- [Colab Setup](#colab-setup)
- [Data](#data)

## Documentation
You can find Project Report about this repository (like journey about finding the best approach) [here](https://drive.google.com/file/d/1aiM3wRbJArVYTSDcNLeqvihDHivxa1hS/view?usp=sharing)

## Description
This project aims to improve captioning for image changes by combining reinforcement learning techniques. Traditional image captioning models often suffer from subtle differences between images, and this project addresses these limitations by introducing reinforcement learning methods that improve and extend the limits of the model’s performance. Our approach consists of two parts: first, we use convolutional neural networks (CNNs) which is ResNet101 to extract and analyze features in before-and-after satellite images, and then we use natural language processing (NLP) techniques such as transformer models to processed and interpreted text. data. A specially designed model that combines the functions of both domains is used to integrate the two data streams and guarantee a thorough understanding of the multimodal data. Reinforcement learning models use pre-trained image caption models as a framework to integrate reinforcement learning to continuously improve captioning features. A reward system was introduced to rate subtitles based on user feedback. This allows the model to create more accurate and contextual descriptions. To train the model, a diverse dataset of pairs of images with captions was compiled. The dashboard allows you to easily collect feedback on the quality of user-generated subtitles and titles to iteratively improve model performance.

## Colab Setup
1. Follow these steps to run the project on Colab: [Colab Setup](https://colab.research.google.com/drive/1FrBvXsUr0ulVuVHl8_fSAlhy9Hf7-GFf?usp=sharing).
2. All processes are explained on the notebook. You can find a couple of approach for implementing our goal.

## Data
1. Download all data from [this link](https://captain-whu.github.io/SCD/).
2. This data is well-structered about labeled which places is changed and what is the changes (like planting trees or buildings built).
3. Our human feed-back points has given for RSICC model's output sentences. You can find the repo [here](https://github.com/Chen-Yang-Liu/RSICC).


