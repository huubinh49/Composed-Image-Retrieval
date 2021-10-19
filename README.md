# Composed-Image-Retrieval
This reposity contain implementation of our research in Composed Image Retrieval task, we use the same train & test protocol as [ComposeAE]{https://github.com/ecom-research/ComposeAE}

## Introduction
Image retrieval systems have seen considerable advances in the last decade, with progress primarily focuses on text-based and content-based methods. However, the inputs to these systems are mainly uni-model, queries are expressed as either texts or images, leaving little to no works are done in the situation of multi-model inputs. This work considers the problem of retrieving images where the input is a combination of texts and images.

![Example of Composed Image Retrieval on FashionIQ dataset]{img/fashioniqsamples.png}
![Example of Composed Image Retrieval on MITState dataset]{img/mitstatesamples.png}
Our model is composed of FiLM module and Stack Attention Network. Its architecture:
![Model Architecture]{img/model.png}

Our paper: 

