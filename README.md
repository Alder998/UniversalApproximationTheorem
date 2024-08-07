## Universal Approximation Theorem, Exploration and Application to Financial Markets ##

When I first approched Statistical Modelling, in 2021, I wanted to put it into practice, and therefore, I created a small Jupyter Program, whose
main purpose was to try to approximate the Empirical Distribution of any traded stocks with combination of known distributions.<br>
The way I set the project that time was to take a huge amount of random observations, and use them as mean and standard deviation for a Mixture
Distribution composed by Normal distributions. <br> <br>

Now, the problem of Stocks returns Distribution Approximation has been taken on from a Deep Learning perspective, using Universal Approximation
Theorem, that is the mathematical foundation of Neural Network. Firstly, I had some fun implementing the structure to approximate known
functions and Distributions. <br>
Then, the core project has been developed. My aim was to **equip a single-Layer Neural Network with known distributions**, and allow it to approximate the distribution of the returns
of (virtually) any traded stocks (from yfinance library), using only Normal Mixture distributions. With this simple exercise, it will be possible to take on **unknown distributions**, and treat them as **combinations
of known functions**. <br><br>
Personally, I think that this procedure could be used to compute more precise and customized VARs (or other statistical Risk Measures) for any single stock. <br><br>

From a Technical Point of view, from this quick project I learned some useful techniques to manage, modify, and custom TesorFlow "core" components (as Layer, Weights, Losses).

