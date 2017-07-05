# Variational Autoencoders

Variational Autoencoders (VAEs) were defined in 2013 by Kingma et al [1]. They allow us to create complex generative models of data, and fit them to large datasets. 

The idea is that the data samples x are generated according to some generative process from latent variables z. The joint probability p(x,z) = p(x|z)p(z) decomposes into a prior p(z) of the latent variables and a likelihood p(x|z). Our goal is to find good values for the latent variables given the observed data, or to calculate the posterior p(z|x).

![vae diagram](figures/vae-diagram.png)

In a VAE we will approximate the true posterior p(z|x) by a probability q(z|x) represented by an encoder neural network that takes a data point x as input and outputs a distribution q(z|x). A decoder network will take a sample from the latent space as input and output an likelihood distribution p(x|z). To train both neural networks, we will minimize the following loss function:

![vae loss](https://latex.codecogs.com/gif.latex?L_i%28%5Ctheta%2C%5Cphi%29%20%3D%20-E_%7Bz%7Eq_%5Ctheta%28z%7Cx_i%29%7D%5B%5Clog%20p_%5Cphi%28x_i%7Cz%29%5D%20&plus;%20KL%28q_%5Ctheta%28z%7Cx_i%29%7C%7Cp%28z%29%29%29)

The first term is the reconstruction loss, or negative expected log-likelihood of the i-th data point. This will encourage the decoder to learn to reconstruct the input data from the latent variables. The second term is a regularizer, the Kullback-Leibler divergence between the encoder's distribution q(z|x) and the prior p(z). In a VAE the prior p(z) is specified as a standard Normal distribution with mean zero and variance one.

To let our neural networks output probability distributions while still being able to backpropagate, we let the encoder and decoder output means and variances of a multivariate gaussian distribution. Using the reparameterization trick we can then sample from these distributions and backpropagate to the means and variances.


## VAE in DIANNE

In order to train a VAE in DIANNE, you can use the VariationalAutoEncoderLearningStrategy. This strategy requires two neural networks, the encoder and decoder, and the dimension of the latent space. 

To run this example, make sure you have the MNIST dataset in the datasets folder. You can download the MNIST dataset by executing the following build command in the root directory:
```
./gradlew datasets -Pwhich=MNIST
```

As Encoder network, we use a 3 layer fully connected neural network with 512 hidden neurons in each layer. We will encode each sample to a 2 dimensional space, so our output is a 2d multivariate gaussian. Hence, the last Linear layer ends with 4 outputs, 2 means and 2 stdevs of the distribution.

Similarly, we use a 3 layer fully connected neural network as Decoder. This neural net takes as input a latent sample of dimension 2, and outputs a probability for each pixels to be black or white. We reshape to a 28x28 image for visualization.

Now we can train the networks using the VariationalAutoEncoderLearningStrategy using the following command: 

```
dianne:learn Encoder,Decoder MNIST autoencode=true range=0,59999 strategy=VariationalAutoEncoderLearningStrategy criterion=BCE latentDims=2 method=ADAM batchSize=128 sampleSize=1 maxIterations=5000 trace=true traceInterval=1 tag=vae
```

The command explained:

* dianne:learn : we start a learn job
* Encoder,Decoder : our learning strategy needs two neural net instances: an encoder and a decoder network
* MNIST : we use the MNIST dataset to get the data 
* autoencode=true : instead of the MNIST labels, we want to use the input sample as target instead of the label
* range=0,59999 : we only use the train set (first 60000 samples) 
* strategy=VariationalAutoEncoderLearningStrategy  : our VAE strategy
* criterion=BCE : the decoder loss is Binary Cross Entropy loss since a pixel can be either black or white in the dataset
* latentDims=2 : we encode to 2 latent dimensions
* method=ADAM : use the ADAM optimizer
* batchSize=128 : use a minibatch size of 128
* sampleSize=1 : in each training iteration, sample only once from the Encoder distribution
* maxIterations=5000 : after 5000 iterations one should already see some valid results
* trace=true : trace intermediate output to the command line
* traceInterval=1 : log progress for each iteration
* tag=vae : use a separate tag for storing/retrieving the weights

Once trained, you can check whether it works by deploying an instance of your Decoder (make sure to reuse the same tag as you trained with, i.e. "vae"), and forward some random data through it. The Decoder should generate valid reconstructions from the MNIST dataset. You can also forward custom latent dimensions to inspect how the VAE mapped the data points into the latent space. 

![vae](figures/vae.png)


## References

[[1]](https://arxiv.org/abs/1312.6114) Diederik P Kingma, Max Welling, Auto-Encoding Variational Bayes.