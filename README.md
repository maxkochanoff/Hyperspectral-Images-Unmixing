Hyperspectral imaging is a specific type of image data acquisition that uses several wavelength sensors (from few to hundreds) to extract qualitative information from each pixel of the seen distant object. The main application frame assumes the picture being composed out of a fixed number of materials (called endmembers or sources), each one characterized by a unique specter, defined on the channel range. The final objective is to describe the observed picture through the proportions (called abundances) of every material within every pixel. This process of transforming the rough observation X into endmembers matrix A and abundance matrix S is called unmixing.

As in most image processing tasks, two approaches are available : algorithmic one and network one, and each corresponds to a conceptual viewpoint on the problem. Generally, man can mathematically describe the situation and make assumptions about the matrices X, A, S in order to solve X = AS as precisely as possible. We can resolve it with for example the PALM algorithm. On the other hand, man can see unmixing as an application of the information compression task. Indeed, the goal is to simplify the pixel description from hundreds of channels to a few aboundances. And this corresponds to the compact data representation of autoencoder latent space.




All in all, the study pipeline can be described in two parts:

**Part 1: Perturbations Approach**

1. Getting the First Estimation: Denoting X as the hyperspectral image to unmix, the first step consists of applying an unsupervised unmixing algorithm to provide the initial estimate of A∗ and S∗. In this study, we utilize a neural approach with an Autoencoder.

2. Endmember Perturbations: In this stage, we compare two different approaches:

  * A random perturbation function is applied N times to each endmember in A∗ to generate realistic data augmentation of the extracted endmembers.

  * A pixel-wise perturbation is applied to the generated training images to better mimic real-world hyperspectral images.

3. Autoencoder Training on Corresponding Data

**Part 2: Variational Autoencoder (VAE) Approach**

1. Constructing a Training Dataset for a VAE: By utilizing the original image, we can obtain approximations of endmembers as a result of the Vertex Component Analysis (VCA) algorithm's operation.

2. Generation of Endmembers: Having obtained several examples of endmember approximations, we train the VAE. The subsequently generated endmembers via VAE can also be considered somewhat perturbed, as the training dataset consisted of slightly differing approximations of the same endmembers.

3. Autoencoder Training on Corresponding Data: Abundances were obtained using Fully Constrained Least Squares (FCLS), followed by the acquisition of the hyperspectral image using a linear model.



To explore more about the project and the results, you can check the attached Project Report and notebooks.

<p align="center">
  <img width="621" alt="image" src="https://github.com/maxkochanoff/Hyperspectral-Images-Unmixing/assets/122701199/e15a457e-500e-465a-8ea0-b7dda99e0cb8">
</p>

