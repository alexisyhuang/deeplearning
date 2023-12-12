---
layout: distill
title: "Autodecoders: Analyzing the Necessity of Explicit Encoders in Generative Modeling"
description: The traditional autoencoder architecture consists of an encoder and a decoder, the former of which compresses the input into a low-dimensional latent code representation, while the latter aims to reconstruct the original input from the latent code. However, the autodecoder architecture skips the encoding step altogether and trains randomly initialized latent codes per sample along with the decoder weights instead. We aim to test the two architectures on practical generative tasks as well as dive into the theory of autodecoders and why they work along with their benefits.
date: 2023-12-11
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Seok Kim
    affiliations:
      name: Massachusetts Institute of Technology
  - name: Alexis Huang
    affiliations:
      name: Massachusetts Institute of Technology

# must be the exact same name as your blogpost
bibliography: 2023-11-09-autodecoders.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Related Works
  - name: Methodology
  - name: Conclusion
  - name: References

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Autodecoders

### Introduction

Autoencoders have been a part of the neural network landscape for decades, first proposed by LeCun in 1987. Today, many variants of the autoencoder architecture exist as successful applications in different fields, including computer vision and natural language processing, and the autoencoder remains at the forefront of generative modeling. Autoencoders are neural networks that are trained to reconstruct their input as their output, accomplishing this task with the use of an encoder-decoder network.

Autoencoders comprise of the encoder network, which takes a data sample input and translates it to a lower-dimensional latent representation, and the decoder network, which reconstructs the data from this encoding. By learning a compressed, distributed representation of the data, autoencoders greatly assist with dimensionality reduction.

With traditional autoencoders, the encoder-decoder network is trained, but only the decoder is retained for inference. Because the encoder is not used at test time, training an encoder may not be an effective use of computational resources; the autodecoder is an alternative architecture that operates without an encoder network.

Rather than using the encoder to encode the input into a low-dimensional latent code, each sample in the training set begins with a randomly initialized latent code, and the latent codes and the decoder weights are both updated during training time. For inference on new data, the latent vector for a given sample is also randomly initialized and updated through an additional optimization loop.

{% include figure.html path="/assets/img/2023-12-11-autodecoders/encoderdecoder.png" class="img-fluid" style="display:block;"%}

### Related Works

The Generative Latent Optimization framework was introduced by Bojanowski et al. (2019) as an alternative to the adversarial training protocol of GANs. Instead of producing the latent representation with a parametric encoder, the representation is learned freely in a non-parametric manner. One noise vector is optimized by minimizing a simple reconstruction loss and is mapped to each image in the dataset.

Tang, Sennrich, and Nivre (2019) trained encoder-free neural machine translation (NMT) models in an endeavour to produce more interpretable models. In the encoder-free model, the source was the sum of the word embeddings and the sinusoid embeddings (Vaswani et al., 2017), and the decoder was a transformer or RNN. The models without an encoder produced significantly poorer results; however, the word embeddings produced by encoder-free models were competitive with those produced by the default NMT models.

DeepSDF, a learned continuous Signed Distance Function (SDF) representation of a class of shapes, was introduced by Park et al. (2019) as a novel representation for generative 3D modeling. Autodecoder networks were used for learning the shape embeddings, trained with self-reconstruction loss on decoder-only architectures. These autodecoders simultaneously optimized the latent vectors mapping to each data point and the decoder weights through backpropagation. While outperforming previous methods in both space representation and completion tasks, autodecoding was significantly more time-consuming during inference because of the explicit need for optimization over the latent vector.

Sitzmann et al. (2022) introduced a novel neural scene representation called Light Field Networks (LFNs), reducing the time and memory complexity of storing 360-degree light fields and enabling real-time rendering. 3D scenes are individually represented by their individual latent vectors that are obtained by using an autodecoder framework, but it is noted that this may not be the framework that performs the best. The latent parameters and the hypernetwork parameters are both optimized in the training loop using gradient descent; the LFN is conditioned on a single latent variable. Potential applications are noted to include enabling out-of-distribution through combining LFNs with local conditioning.

Scene Representation Networks (SRNs) represent scenes as continuous functions without knowledge of depth or shape, allowing for generalization and applications including few-shot reconstruction. SRNs, introduced by Sitzmann, Zollhöfer, and Wetzstein (2019), represent both the geometry and appearance of a scene, and are able to accomplish tasks such as novel view synthesis and shape interpolation from unsupervised training on sets of 2D images. An autodecoder framework is used to find the latent vectors that characterize the different shapes and appearance properties of scenes.

### Methodology

#### Traditional Autoencoder

To establish a baseline, we first trained a convolutional autoencoder network containing both an encoder and decoder on a version of the MNIST dataset normalized and padded to contain 32x32 images. For our autoencoder architecture, we  utilized convolutional layers with ReLU nonlinearity.
{% include figure.html path="/assets/img/2023-12-11-autodecoders/autoencoderloss.png" class="img-fluid" caption="The training and validation losses from the training loop for the autoencoder."%}
{% include figure.html path="/assets/img/2023-12-11-autodecoders/tsne_autodecoder.png" class="img-fluid" caption="The training and validation losses from the training loop for the autoencoder." caption="The latent space learned by the autoencoder, color-coded by digit label and visualized through a 2-dimensional t-SNE plot. We see the expected result, with consistency and separation."%}
{% include figure.html path="/assets/img/2023-12-11-autodecoders/autoencoderloss.png" class="img-fluid" caption="The training and validation losses from the training loop for the autoencoder." caption="A sample output from an unseen image after training. We can see that our small convolutional autoencoder does a fairly good job at learning how to compress simple information into a single latent code."%}

#### Autodecoder

We implemented and trained an autodecoder on the same dataset by creating a convolutional decoder that takes latent codes as an input and upscales them to a full image. We utilized transpose convolutions to upscale the images while additionally concatenating normalized coordinates to include positional information, along with leaky ReLU layers for nonlinearity.

For training, the latent codes for 10,000 images in our training set were randomly initialized. The loss for our autodecoder then included three components: the reconstruction loss; the latent loss, which encourages latent values to be closer to zero in order to encourage a compact latent space; and the L2 weight regularization, which prevents the decoder from overfitting to the training set by encouraging the model weights to be sparse. 
{% include figure.html path="/assets/img/2023-12-11-autodecoders/autodecoderloss.png" class="img-fluid" caption="The training and validation losses from the training loop for the autoencoder." caption="The training and validation losses from the training loop for the autodecoder. The validation loss has no actual meaning in the autodecoder framework, as new images would have a randomly initialized latent code, and was included simply to demonstrate this feature."%}

Below are progressive reconstructions on the training data performed by the autodecoder as it trained and optimized both the decoder weights and training set’s latent codes. We observe that the digits’ general forms were learned before the exact shapes, which implies good concentration and consistency of the latent space between digits of the same class.

{% include figure.html path="/assets/img/2023-12-11-autodecoders/progress1.png" class="img-fluid"%}
{% include figure.html path="/assets/img/2023-12-11-autodecoders/progress2.png" class="img-fluid"%}
{% include figure.html path="/assets/img/2023-12-11-autodecoders/progress3.png" class="img-fluid"%}
{% include figure.html path="/assets/img/2023-12-11-autodecoders/progress4.png" class="img-fluid" caption="Progressive reconstructions  from top to bottom (model outputs compared to ground truth): 1. Decoding a randomly initialized latent code outputs nonsense. 2. The correct digit is reconstructed, implying that the latent space is improving, but the specific shape differs from that of the ground truth image. 3. The output’s shape begins to better match that of the ground truth. 4. The autodecoder and latent code are optimized to be able to effectively reconstruct the ground truth image."%}

{% include figure.html path="/assets/img/2023-12-11-autodecoders/tsne_autodecoder.png" class="img-fluid" caption="The latent space learned by the autodecoder, also visualized through a 2-dimensional t-SNE plot. We again see consistency, but notice that the clusters are more compact. While distance between clusters in  t-SNE plots do not have a definite meaning, this could potentially imply that features of shapes, rather than the shapes themselves, are better learned, as different digits share similar features (curves, straight lines, etc)."%}

Upon training the autodecoder, for inference on a new image we first freeze the decoder weights and then run an additional optimization loop over a randomly initialized latent code.

{% include figure.html path="/assets/img/2023-12-11-autodecoders/autodecodersampleoutput.png" class="img-fluid" caption="Output from the trained autodecoder on a new image from the test set"%} 


### Plan

We will start by providing a detailed overview of how autodecoders function in a comprehensive blog post. This will include a thorough explanation of their architecture, training process, and potential applications. We will also discuss the theoretical advantages and disadvantages of autodecoder networks compared to traditional autoencoders.

Then, for the experimental part of our project, we will construct simple versions of both an autoencoder and an autodecoder network. These networks will be similarly trained and evaluated on a common dataset, such as the widely-used MNIST dataset, where we will attempt to generate novel images with both models. We will then conduct a comparative analysis of the performance of the two different networks, highlighting the differences in their performances and their respective strengths and weaknesses. This experiment will give us a good idea of the efficacy of the two different networks as well as how they compare to each other.

Additionally, we plan to assess whether one network performs better on out-of-distribution generalization tasks. By understanding the potential benefits and drawbacks of autodecoder networks, we can better leverage this innovative approach for a variety of generative tasks and gain insight into their applicability in a broader context.

### References

https://www.inovex.de/de/blog/introduction-to-neural-fields/

https://arxiv.org/pdf/1901.05103.pdf

https://karan3-zoh.medium.com/paper-summary-deepsdf-learning-continuous-signed-distance-functions-for-shape-representation-147af4740485
