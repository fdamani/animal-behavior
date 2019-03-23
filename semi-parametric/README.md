Gameplan:

- fit GLM (linear dynamics prior with fixed scale) using VI with reparam trick.
	- show we can recover dynamics
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