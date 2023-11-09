---
layout: distill
title: "Autodecoders: Analyzing the Necessity of Explicit Encoders in Generative Modeling"
description: The traditional autoencoder architecture consists of an encoder and a decoder, the former of which compresses the input into a low-dimensional latent code representation, while the latter aims to reconstruct the original input from the latent code. However, the recently-developed autodecoder architecture skips the encoding step altogether and trains randomly initialized latent codes per sample instead. We aim to test the two architectures on practical generative tasks as well as dive into the theory of autodecoders and why they work along with their benefits.
date: 2023-11-06
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
  - name: Equations
  - name: Images and Figures
    subsections:
      - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Layouts
  - name: Other Typography?

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

## Project Proposal



### Outline 

For our project, we are planning to investigate the autodecoder network for generative modeling and its benefits and drawbacks when compared to the traditional autoencoder network. We will also explore the potential applications of autodecoders in various domains, particularly in 3D scene reconstruction.

### Background

Autoencoders have been extensively used in representation learning, comprising of the encoder network, which takes a data sample input and translates it to a lower-dimensional latent representation, and the decoder network, which reconstructs the data from this encoding. By learning a compressed, distributed representation of the data, autoencoders greatly assist with dimensionality reduction.

In contrast, the autodecoder network operates without an encoder network for learning latent codes. Rather than using the encoder to encode the input into a low-dimensional latent code, each sample in the training set starts with a randomly initialized latent code, and the latent codes and the decoder weights are both updated during the training time. For inference, the latent vector for a given sample is determined through an additional optimization loop.

```markdown
{% raw %}{% include figure.html path="/assets/img/2023-11-06/autoencoder_schematic.png" class="img-fluid" %}{% endraw %}
```
![mnsit with autoencoder](/assets/img/2023-11-06/autoencoder_schematic.png)

### Applications
One notable application of autodecoder networks is in 3D scene reconstruction. Traditional autoencoders tend to learn a single global latent code, making them less suitable for scenes with multiple objects and complex compositional structures. On the other hand, autodecoders can learn local latent codes, allowing for more efficient performance on scenes with multiple objects. This is particularly valuable in computer vision tasks to understand and reconstruct complex scenes. 

### Plan
We will start by providing a detailed overview of how autodecoders function in a comprehensive blog post. This will include a thorough explanation of their architecture, training process, and potential applications. We will also discuss the theoretical advantages and disadvantages of autodecoder networks compared to traditional autoencoders. Then, for the experimental part of our project, we would we will construct both an autoencoder and an autodecoder network. These networks will be trained and evaluated on a common dataset, likely the widely-used MNIST dataset. We would then conduct a comparative analysis of the performance of the two different networks, highlighting the differences in their performances and their respective strengths and weaknesses. This experiment would give us a good idea of the efficacy of the two different networks as well as how they compare to each other. Additionally, we plan to assess whether one network performs better on out-of-distribution generalization tasks. By understanding the potential benefits and drawbacks of autodecoder networks, we can better leverage this innovative approach for specific tasks, such as 3D scene reconstruction, and gain insights into their applicability in a broader context.

### References
https://www.researchgate.net/figure/Schematic-overview-of-autoencoder-and-autodecoder-architectures-Figure-adapted-from-Park_fig2_356282096
https://www.inovex.de/de/blog/introduction-to-neural-fields/
https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf
https://karan3-zoh.medium.com/paper-summary-deepsdf-learning-continuous-signed-distance-functions-for-shape-representation-147af4740485
https://drive.google.com/file/d/1L3vVWbjcs1TFOc_BasQX43MKa64-dIwa/view


