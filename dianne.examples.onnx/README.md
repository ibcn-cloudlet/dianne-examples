# ONNX neural network format

This example shows how import a model trained with PyTorch using the ONNX format and export it as a .jar file.

## Training with PyTorch

In this project there is a `mnist.py` script, which contains an example PyTorch script to train a simple Convolutional Neural Network (CNN) to classify MNIST digits. To execute this script, PyTorch version 1.0 has to be installed.

```
python mnist.py
```

After training the resulting model is saved in the ONNX format as `mnist.onnx`.


## Importing the .onnx file into DIANNE

Next, launch the `onnx.bndrun` file and import the .onnx file by running 

```
g! dianne:fromOnnx mnist.onnx MNIST
```

on the Gogo shell. This will read in the .onnx file and create a DIANNE model. From the DIANNE builder UI you can now load, deploy and inspect the MNIST model.


## Exporting to a .jar file
 
Finally, the MNIST model can also be exported to a .jar file, adding additional properties of the model, such as the input and output size, as described in the DIANNE namespace. 

```
g! dianne:jar MNIST input.width=28 input.height=28 input=digit output.size=10 output=classification dataset=MNIST version=1.0.0
```

This creates `be.iminds.iot.dianne.nn.mnist.jar` which can be archived inside an OSGi repository.

You can also install and start this .jar inside the running OSGi framework. This should bring online a `NeuralNetwork` service which resolves the component in the example.

```
g! install be.iminds.iot.dianne.nn.mnist.jar
Bundle ID: 51
g! start 51
Neural Network service found!
```

