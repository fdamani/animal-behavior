Gameplan:

- RNN works. problem is its slow
- TODO: for now, train on small dataset. (500 samples).




- fit GLM (linear dynamics prior with fixed scale) using VI with reparam trick. X
- apply to real data. X
- train RNN. (lets train this separately on x to y data.)
	- generate data according to LearningDynamics model
	- fit using RNN.
		- 
- fit RNN on input x_t to predict y_t.
- concat RNN with features to predict.
	- compare elbo.


1. fit RNN on with input x_t and y_{t-1} and predict y_t
2. fit RNN on input x_t and predict y_t. 
	- learn representation H.
3. fit GLM with H + features vs regular GLM 
	- compare ELBO.
4. VI no amortization
5. VI + amortization