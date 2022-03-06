<h1 align="center">Residual Blocks</h1><br>

We can define Neural networks as `Universal function approximators` ($ F(X) = X $) and the accuracy increases with increasing the number of layers. 

But increasing the number of layers return some problems like `vanishing and exploding gradients` and `the curse of dimensionality`, also the accuracy will saturate at one point and eventually degrade. 

If we have sufficiently deep networks, it may not be able to learn even a simple functions like an identity function.


<img src="images/Single Residual Block.png"></img>

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
<a href="https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec">Residual blocks building blocks of Resnet</a><br>
