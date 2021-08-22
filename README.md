# A Matlab implementation of "Extreme Learning Machine: Theory and Applications"

Intro
-----

Let's have a look at Extreme Learning Machines with Matlab (to freshen
up our skills w/ Matlab!)

The Extreme Learning Machine (ELM) is a learning algorithm that first
appeared in [Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew. \"Extreme
learning machine: Theory and applications\", Neurocomputing (70), pp.
489 - 501,
2006](https://www.ntu.edu.sg/home/egbhuang/pdf/ELM-NC-2006.pdf) for
single-hidden layer feedforward neural networks, which randomly chooses
hidden nodes and analytically determines the output weights of the
neural network. It has been reported to show good generalization
performance at extremely fast learning speeds. Here, we'll implement the
training algorithm as presented in the above paper, and also try it on
one of the referenced datasets, just to get familiar with the concept:

Implementation
--------------

### Dummy dataset

At first, let's generate a dummy dataset, to illustrate the training and
prediction process. We will use the same example as in the reference in
the beginning, i.e. we will approximate the 'sinC' function:

$y\left(x\right)=\left\lbrace \begin{array}{ll}
\frac{\sin \left(x\right)}{x} & x\not= 0\\
1 & x=0
\end{array}\right.$,

and to make the regression problem 'real', we'll add uniform noise
distributed in $\left\lbrack -0\ldotp 2,0\ldotp 2\right\rbrack$, while
keeping the testing data, noise-free.

rng(0,'twister');

noSamples = 5e3; x = linspace(-10, 10, noSamples)'; ySinc =
NaN(size(x)); ySinc(x  = 0) = sin(x) ./ x; ySinc(x == 0) = 1;

a = -0.2; b = 0.2; noise = a + (b - a) \* rand(size(ySinc)); y = ySinc +
noise;

figure scatter(x, y, '.', 'displayname', 'sinC w/ noise')

Warning: MATLAB has disabled some advanced graphics rendering features
by switching to software OpenGL. For more information, click here. hold
on plot(x, ySinc, 'k', 'linewidth', 3, 'displayname', 'sinC')
xlabel('x') ylabel('y') title('sinC function') legend

![](/ELM_images/figure_0.eps)

We'll use 70% of the data, chosen randomly, as the training set, to
which we'll add noise, and the rest 30% noise-free data will be used as
the test set:

nTrain = floor(0.7 \* noSamples);

idxTrain = randi(\[1, noSamples\], nTrain, 1); idxTest =
setdiff(1:noSamples, idxTrain);

xTrain = x(idxTrain); yTrain = y(idxTrain); xTest = x(idxTest); yTest =
ySinc(idxTest);

### **Training Algorithm** {#training-algorithm .unnumbered}

The training algorithm consists of 3, rather simple steps, as discussed
in the following.

At first, let's define some necessary constants:

Ntilde = 30; N = size(yTrain, 1); n = size(xTrain, 2); m = size(yTrain,
2);


#### **Step 1: Randomly assign input weight vector** ${\mathit{\mathbf{w}}}_i ={\left\lbrack w_{\textrm{i1}} ,w_{\textrm{i2}} ,\ldotp \ldotp \ldotp ,w_{\textrm{in}} \right\rbrack }^T \in \Re^n$ **and bias** ${\mathit{\mathbf{b}}}_i ={\left\lbrack b_{\textrm{i1}} ,b_{\textrm{i2}} ,\ldotp \ldotp \ldotp ,b_{\textrm{im}} \right\rbrack }^T \in \Re^m ,i=1,\cdots ,N$ {#step-1-randomly-assign-input-weight-vector-mathitmathbfw_i-leftlbrack-w_textrmi1-w_textrmi2-ldotp-ldotp-ldotp-w_textrmin-rightrbrack-t-in-ren-and-bias-mathitmathbfb_i-leftlbrack-b_textrmi1-b_textrmi2-ldotp-ldotp-ldotp-b_textrmim-rightrbrack-t-in-rem-i1cdots-n .unnumbered}

w = rand(\[n, Ntilde\]); b = rand(\[m, Ntilde\]);

[\[H_AF88CEAB\]]{#H_AF88CEAB label="H_AF88CEAB"}

\>0

#### **Step 2: Compute the hidden layer output matrix H** {#step-2-compute-the-hidden-layer-output-matrix-h .unnumbered}

$$\mathit{\mathbf{H}}\left(w_1 ,\ldotp \ldotp \ldotp ,w_{\tilde{N} } ,b_1 ,\ldotp \ldotp \ldotp ,b_{\tilde{N} } ,x_1 ,\ldotp \ldotp \ldotp ,x_N \right)={\left\lbrack \begin{array}{ccc}
g\left(w_{1\;} x_1 +b_1 \right) & \cdots  & g\left(w_{\tilde{N} } x_1 +b_{\tilde{N} } \right)\\
\vdots  & \ddots  & \vdots \\
g\left(w_{1\;} x_N +b_1 \right) & \cdots  & g\left(w_{\tilde{N} } x_N +b_{\tilde{N} } \right)
\end{array}\right\rbrack }_{N\times \tilde{N} }$$

where:

${\mathit{\mathbf{x}}}_i ={\left\lbrack x_{\textrm{i1}} ,x_{\textrm{i2}} ,\ldotp \ldotp \ldotp ,x_{\textrm{in}} \right\rbrack }^T \in \Re^n ,{\mathit{\mathbf{w}}}_i ={\left\lbrack w_{\textrm{i1}} ,w_{\textrm{i2}} ,\ldotp \ldotp \ldotp ,w_{\textrm{in}} \right\rbrack }^T \in \Re^n$,
and g() is the activation function. As proposed in the paper, we'll use
the sigmoid function.

H = sigm(xTrain \* w + b);

[\[H_51A560F0\]]{#H_51A560F0 label="H_51A560F0"}

\>0

#### **Step 3: Calculate the output weight** $\beta$ {#step-3-calculate-the-output-weight-beta .unnumbered}

$\beta ={\mathit{\mathbf{H}}}^{\dagger } \;\mathit{\mathbf{Y}}$, where
${\mathit{\mathbf{H}}}^{\dagger }$ is the Moore-Penrose generalized
inverse of matrix H, given by:
${\mathit{\mathbf{H}}}^{\dagger \;} ={\left({\mathit{\mathbf{H}}}^{\mathit{\mathbf{T}}\;} \mathit{\mathbf{H}}\right)}^{-1} {\mathit{\mathbf{H}}}^{\mathit{\mathbf{T}}\;} \ldotp$

Here, we will use the computational procedure of [Fast Computation of
Moore-Penrose Inverse Matrices, Pierre
Courrieu](https://arxiv.org/ftp/arxiv/papers/0804/0804.4809.pdf), based
on a full rank Cholesky factorization. Implementation can be seen at the
end of the notebook, in the pseudo_inv() function.

beta = pseudoInv(H) \* yTrain;

And that's it. We just learned the neural network.

[\[H_AFF803AD\]]{#H_AFF803AD label="H_AFF803AD"}

\>0

### **Making Predictions** {#making-predictions .unnumbered}

Making predictions is a simple one-liner. Let's predict on the
noise-free test set, and plot the results:

yHat = sigm(xTest \* w + b) \* beta;

figure scatter(xTest, yTest, 'displayname', 'actual') hold on
scatter(xTest, yHat, 'x', 'displayname', 'predicted') legend ylabel('y')
xlabel('x') title('test set performance')

![image](figure_1.eps){width="\\ifdim\\linewidth>56.196688409433015em 56.196688409433015em\\else\\linewidth\\fi"}

It is clear that the ELM can succesfully predict the sinc function.

[\[H_BE04E1C1\]]{#H_BE04E1C1 label="H_BE04E1C1"}

\>0

### Sensitivity to number of hidden neurons {#sensitivity-to-number-of-hidden-neurons .unnumbered}

It will be fairly easy to visualize the performance (root mean square
error) of the ELM with a different number of hidden neurons. Let's wrap
the above procedure into a few functions for the training and
predictions processes (train() and predict() as seen in the end,
respectively), and plot performance:

rng(1,'twister');

noNeurons = 5:5:100; rmse = NaN(size(noNeurons));

for i = 1:length(noNeurons) Ntilde = noNeurons(i); \[w, b, beta\] =
train(xTrain, yTrain, Ntilde); yHat = predict(xTest, w, b, beta);
rmse(i) = mean(sqrt((yTest - yHat) .\^2)); end

figure plot(noNeurons, rmse, '-o') xlabel('No. neurons in hidden layer')
ylabel('RMSE - test set') title('Sensitivity to no. of neurons')

![image](figure_2.eps){width="\\ifdim\\linewidth>56.196688409433015em 56.196688409433015em\\else\\linewidth\\fi"}

Indeed, as the paper reports, the ELM is stable on a wide range of
number of hidden neurons, and equally importantly, super-fast!

[\[H_F9AB4FE2\]]{#H_F9AB4FE2 label="H_F9AB4FE2"}

\>0

Trial on a more realistic dataset {#trial-on-a-more-realistic-dataset .unnumbered}
---------------------------------

Let's check the performance on a real dataset. We'll use the [Delta
elevators](https://www.dcc.fc.up.pt/~ltorgo/Regression/delta_elevators.html)
dataset which contains 9517 instances of 6 attributes. It is obtained
from the task of controlling the elevators of a F16 aircraft, and its a
regression task.

[\[H_E104AB0A\]]{#H_E104AB0A label="H_E104AB0A"}

\>0

### Quick Preprocessing {#quick-preprocessing .unnumbered}

Let's prepare the dataset for the ELM. We'll do a simple train/test
split with the normalization process described in the initial paper:

clear all

dset = readtable('deltaElevators.txt'); dset = table2array(dset);

x = dset(:, 1:6); y = dset(:, 7);

for col = 1:size(x, 2) mmax = max(x(:, col)); mmin = min(x(:, col));
range = mmax - mmin;

x(:, col) = (x(:, col) - mmin) / range; end

for col = 1:size(y, 2) mmax = max(y(:, col)); mmin = min(y(:, col));
range = mmax - mmin; y(:, col) = 2 \* (y(:, col) - mmin) / range - 1;
end

noSamples = size(x, 1); nTrain = floor(0.5 \* noSamples);

idxTrain = randi(\[1, noSamples\], nTrain, 1); idxTest =
setdiff(1:noSamples, idxTrain);

xTrain = x(idxTrain, :); yTrain = y(idxTrain, :); xTest = x(idxTest, :);
yTest = y(idxTest, :);

[\[H_17854A98\]]{#H_17854A98 label="H_17854A98"}

\>0

### Training and test-set performance {#training-and-test-set-performance .unnumbered}

Let's train the network and get the performance on the test set:

rng(0,'twister');

noNeurons = 5:5:200; noTargets = size(yTrain, 2); rmse =
NaN(length(noNeurons), noTargets);

for i = 1:length(noNeurons) Ntilde = noNeurons(i); \[w, b, beta\] =
train(xTrain, yTrain, Ntilde); yHat = predict(xTest, w, b, beta);
rmse(i, :) = mean(sqrt((yTest - yHat) .\^2)); end

figure plot(noNeurons, rmse, '-o') xlabel('No. neurons in hidden layer')
ylabel('RMSE - test set') title('Sensitivity to no. of neurons')

![image](figure_3.eps){width="\\ifdim\\linewidth>56.196688409433015em 56.196688409433015em\\else\\linewidth\\fi"}

Once again, the RMSE decreases, flattering to about 0.080 after 6
neurons. In the original paper, the authors reported an RMSE of 0.066 on
their test set with 125 hidden neurons. The most remarkable attribute is
the training speed (which is on average constant here - probably due to
the small number of neurons). Note that most of the training process is
spent to compute the Moore-Penrose pseudoinverse matrix.

[\[H_9B2BD61F\]]{#H_9B2BD61F label="H_9B2BD61F"}

\>0

Function definitions {#function-definitions .unnumbered}
--------------------

In the following, we declare all the functions mentioned in the above
sections.

[\[H_7B464DF9\]]{#H_7B464DF9 label="H_7B464DF9"}

\>0

### Sigmoid activation function {#sigmoid-activation-function .unnumbered}

function S = sigm(x)

S = 1./ (1 + exp(-x)); end

[\[H_5561D5DB\]]{#H_5561D5DB label="H_5561D5DB"}

\>0

### Moore - Penrose inverse {#moore---penrose-inverse .unnumbered}

function Y = pseudoInv(G)

\[m, n\] = size(G); transpose = false;

if m \< n transpose = true; A = G \* G'; n = m; else A = G' \* G; end

dA = diag(A); tol = min(dA(dA \> 0)) \* 1e-9; L = zeros(size(A)); r = 0;

for k=1:n r = r + 1; L(k:n, r) = A(k:n, k) - L(k:n, 1:(r - 1)) \* L(k,
1:(r - 1))';

if L(k,r) \> tol L(k,r) = sqrt(L(k, r)); if k\<n L((k + 1):n, r) = L((k
+ 1):n, r) / L(k, r); end else r = r - 1; end end L = L(:, 1:r);

M = inv(L' \* L); if transpose Y = G' \* L \* M \* M \* L'; else Y = L
\* M \* M \* L' \* G'; end end

[\[H_A1D0A95D\]]{#H_A1D0A95D label="H_A1D0A95D"}

\>0

### Training process {#training-process .unnumbered}

function \[w, b, beta\] = train(X, y, noNeurons)

n = size(X, 2); m = size(y, 2);

w = rand(\[n, noNeurons\]); b = rand(\[m, noNeurons\]); beta =
NaN(\[noNeurons, m\]);

for i = 1:m H = sigm(X \* w + b(i, :));

beta(:, i) = pseudoInv(H) \* y(:, i); end end

[\[H_F50AC2AE\]]{#H_F50AC2AE label="H_F50AC2AE"}

\>0

### Prediction

function yHat = predict(X, w, b, beta)

m = size(b, 1); N = size(X, 1); yHat = NaN(N, m);

for i = 1:m yHat(:, i) = sigm(X \* w + b(i, :)) \* beta(:, i); end end

