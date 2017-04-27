# Variational Autoencoders

Variational Autoencoders (VAEs) were defined in 2013 by Kingma et al [1]. They allow us to create complex generative models of data, and fit them to large datasets. 

The idea is that the data samples x are generated according to some generative process from latent variables z. The joint probability p(x,z) = p(x|z)p(z) decomposes into a prior p(z) of the latent variables and a likelihood p(x|z). Our goal is to find good values for the latent variables given the observed data, or to calculate the posterior p(z|x).

In a VAE we will approximate the true posterior p(z|x) by a probability q(z|x) represented by an encoder neural network that takes a data point x as input and outputs a distribution q(z|x). A decoder network will take a sample from the latent space as input and output an likelihood distribution p(x|z). To train both neural networks, we will minimize the following loss function:

![vae loss](https://latex.codecogs.com/gif.latex?L_i%28%5Ctheta%2C%5Cphi%29%20%3D%20-E_%7Bz%7Eq_%5Ctheta%28z%7Cx_i%29%7D%5B%5Clog%20p_%5Cphi%28x_i%7Cz%29%5D%20&plus;%20KL%28q_%5Ctheta%28z%7Cx_i%29%7C%7Cp%28z%29%29%29)

The first term is the reconstruction loss, or negative expected log-likelihood of the i-th data point. This will encourage the decoder to learn to reconstruct the input data from the latent variables. The second term is a regularizer, the Kullback-Leibler divergence between the encoder's distribution q(z|x) and the prior p(z). In a VAE the prior p(z) is specified as a standard Normal distribution with mean zero and variance one.

To let our neural networks output probability distributions while still being able to backpropagate, we let the encoder and decoder output means and variances of a multivariate gaussian distribution. Using the reparameterization trick we can then sample from these distributions and backpropagate to the means and variances.


## VAE in DIANNE

In order to train a VAE in DIANNE, you can use the VariationalAutoEncoderLearningStrategy. This strategy requires two neural networks, the encoder and decoder, and the dimension of the latent space. 

```
dianne:learn VAE_MNIST_encoder,VAE_MNIST_decoder MNIST autoencode=true range=0,59999 strategy=VariationalAutoEncoderLearningStrategy criterion=BCE latentDims=2 method=ADAM batchSize=128 sampleSize=1 maxIterations=100000 trace=true traceInterval=1
```

```
dianne:eval VAE_MNIST_encoder,VAE_MNIST_decoder MNIST autoencode=true range=60000,69999 strategy=VariationalAutoEncoderEvaluationStrategy criterion=BCE latentDims=2 batchSize=128 includeOutputs=true trace=true
```


[[1]](https://arxiv.org/abs/1312.6114) Diederik P Kingma, Max Welling, Auto-Encoding Variational Bayes.