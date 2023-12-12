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
  - name: Experimentation
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

{% include figure.html path="/assets/img/2023-11-09-autodecoders/autoencoder_schematic.png" class="img-fluid" %}
<div style="display: flex; justify-content: space-between;">
  {% include figure.html path="/assets/img/2023-12-11-autodecoders/progress1.png" class="img-fluid" style="width: 25%;" %}
  {% include figure.html path="/assets/img/2023-12-11-autodecoders/progress2.png" class="img-fluid" style="width: 25%;" %}
  {% include figure.html path="/assets/img/2023-12-11-autodecoders/progress3.png" class="img-fluid" style="width: 25%;" %}
  {% include figure.html path="/assets/img/2023-12-11-autodecoders/progress4.png" class="img-fluid" style="width: 25%;" %}
</div>

### Applications

One notable application of autodecoder networks is in 3D scene reconstructions. Traditional autoencoders tend to learn a single global latent code, making them less suitable for scenes with multiple objects and complex compositional structures. On the other hand, autodecoders can learn local latent codes, allowing for more efficient performance on scenes with multiple objects. This is particularly valuable in inverse graphics tasks to understand and reconstruct novel views of complex scenes.

### Plan

We will start by providing a detailed overview of how autodecoders function in a comprehensive blog post. This will include a thorough explanation of their architecture, training process, and potential applications. We will also discuss the theoretical advantages and disadvantages of autodecoder networks compared to traditional autoencoders.

Then, for the experimental part of our project, we will construct simple versions of both an autoencoder and an autodecoder network. These networks will be similarly trained and evaluated on a common dataset, such as the widely-used MNIST dataset, where we will attempt to generate novel images with both models. We will then conduct a comparative analysis of the performance of the two different networks, highlighting the differences in their performances and their respective strengths and weaknesses. This experiment will give us a good idea of the efficacy of the two different networks as well as how they compare to each other.

Additionally, we plan to assess whether one network performs better on out-of-distribution generalization tasks. By understanding the potential benefits and drawbacks of autodecoder networks, we can better leverage this innovative approach for a variety of generative tasks and gain insight into their applicability in a broader context.

### References

https://www.inovex.de/de/blog/introduction-to-neural-fields/

https://arxiv.org/pdf/1901.05103.pdf

https://karan3-zoh.medium.com/paper-summary-deepsdf-learning-continuous-signed-distance-functions-for-shape-representation-147af4740485
