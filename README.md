<h1 align="center">Residual Blocks</h1><br>

We can define Neural networks as `Universal function approximators` **F(X) = X** and the accuracy increases with increasing the number of layers. 

But increasing the number of layers return some problems like `vanishing and exploding gradients` and `the curse of dimensionality`, also the accuracy will saturate at one point and eventually degrade. 

If we have sufficiently deep networks, it may not be able to learn even a simple functions like an identity function.


<img src="images/Single Residual Block.png" align="center"></img>

The idea behind the above block is, instead of hoping each few stacked layers directly fit a desired underlying mapping say **\(H(x)\)**, we explicitly let these layers fit a residual mapping i.e.. **\(F(x) = H(x) - x\)**. Thus original mapping **\(H(x)\)** becomes **\(F(x) + x\)**.

**Shortcut connections**<br>
These connections are those skipping one or more layers. F(x)+x can be understood as feedforward neural networks with “shortcut connections”.

**Why deep residual framework?**<br>
The idea is motivated by the degradation problem (training error increases as depth increases). Suppose if the added layers can be constructed as identity mappings, a deeper model should have training error no greater than its shallower counterpart.

If identity mappings are optimal, it is easier to make F(x) as 0 than fitting H(x) as x (as suggested by degradation problem).

**Results**<br>
In this section, we see the performance of residual networks. The models were trained on the 1.28 million training images, and evaluated on the 50,000 validation images. The performance of the networks were evaluated using top-1 error rate.

In the below table, two types of network are mentioned.

**Plain Networks.**<br>
The deeper neural networks are constructed by stacking more layers on one another without shortcut connections.

<a href="https://arxiv.org/abs/1512.03385">Paper</a><br>
<a href="https://swethatanamala.github.io/2018/07/09/Summary-of-resnet-paper/">Paper Summary</a><br>


### Residual Block

<img src="images/WideResidualNetwork.png" align="center"></img>

<h3>I'm goinng to implment Basic and Bottleneck Residual blocks only:</h3><br>
    
<li> <b>Basic Block</b>: Includes 2 operations:
    <ul>
        <li> 3x3 convolution with padding followed by BatchNorm and ReLU.
        <li> 3x3 convolution with padding followed by BatchNorm.
    </ul>

<li> <b>Bottleneck Block</b>: include 3 operations:
    <ul>
        <li> 1x1 convolution followed by BatchNorm and ReLU.
        <li> 3x3 convolution with stride followed by BatchNorm and ReLU.
        <li> 1x1 convolution followed by BatchNorm.
    </ul>
    
  
The whole block is called one block (layer), which is consists of multiple layers (Conv, BN, ReLU).<br>
<a href="https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec">Residual blocks building blocks of Resnet</a><br>

  
### Types of Residual Block
<img src="images/Types of Residual Block.png" height="300" width="700" align="center">
  
### Types of  shortcut connections residual neural network

The shortcut connections of a residual neural network can be:
- **An identity block**, which is employed when the input and output have the same dimensions. 
- **A Projection block**, which is a convolution block, used when the dimensions are different, it offers a channel-wise pooling, often called feature map pooling or a projection layer.

<img src="images/The shortcut connections of ResNet.jpg" height="500" width="500" align="center"></img>

<a href="https://www.researchgate.net/publication/339109948_Deep_Residual_Learning_for_Nonlinear_Regression/figures?lo=1">Deep Residual Learning for Nonlinear Regression</a>


> #### I'm goinng to implement the `original` architecture of `Basic Block` & `Bottleneck Block` with both Identity and Projection short-cut connection.
