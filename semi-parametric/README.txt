4/7
	- initialize VI with map estimate. it HELPS!
	- then do joint optimization of vi and model params using SGD (try with momentum)
	- see how brittle the method is to initialization. e.g. for map estimate, condition on wrong
		model params. does vi still convergence reasonably well?
	- once we have this working well, add additional model params.



	- optimize variational parameters and model parameters in an iterative optimization framework
		- condition on reasonable model parameters and update variational params. 
		- condition on variatioanl params and update model parameters.
		- adam seems fine for variational params. check to see if adam converges to good model parameter values
			in iterative framework or do we need to use SGD.
	- open question: how to do this on a single trial level. 

4/6

- compute posterior and standard dev param using VI. (show values recover synthetic data)
- write bootstrap code and get this to work. 


4/5
- add learning component back into model (start w/ no regularization)
- show we can get posterior credible intervals for alpha
- do all of the below just for gaussian noise term. show we can estimate this well across
datasets.
	- simplest/fastest approach
		- optimize elbo to get converged results
		- compute laplace approx of model params and create plots with the diagonal of hessian
		showing uncertainty measurements.
	- repeat this for 10 datasets and plot these credible intervals.
- repeat this except with regularization parameters
	- can you recover simulated estimates? have plots showing results from synthetic data.
	- repeat for multiple datasets
- extend to vector of alphas

- for each of these plot
	- posterior credible intervals using post-hoc laplace approximation.
	- validation performance (held out, held out blocks, and next session)
		- compare posterior predictives. remember its not average of logs its log of averages
			- posterior predictive: p(y|x) = expectation_posterior[likelihood] -> MC estimate is fine make sure you average many samples.
				***make sure you're using the likelihood, not log likelihood.



- how to get uncertainty estimates of model parameters
	- get proper uncertainty - approx inference or bootstrap
	- clever ways of training on different subsets of the data and show its consistent



- for regularization story
	- try with only l2 regularization first.


4/3
in the morning:
	- add learning component
		- show learning component and learning+regularization helps with valid loss
	- fix code to estimate model parameters as well. show we can do this well.
	

4/2
- validation loss
	- leave 10% random samples show we can predict those samples with inferred latents.
	- currently outputting train/test marginal likelihoods () and accuracy measures.
		- reporting test ll should be enough to compare against models.
	- band validation: for any index picked remove a chunk. (5-10 samples)
	- predict last session
		- we have future trajectories
		- not exactly useful if we dont have true latents (dont worry about this yet)

		- compute marginal ll on new trajectory.
			- repeat this procedure with multiple "folds". 
		
		- plot reward dynamics vs true (also create these plots for whole trajectory)
	


TODO:
- plot reward dynamics vs true (also create these plots for whole trajectory)


adding to prior
	- add back learning component + regularization
	- show data simulated according to regularization can be recovered by regularization approach
		- e.g. lowest test loss is for model with regularization as opposed to without. (do this with set model params)
	- if prev step works, show we can accurately estimate all model parameters (and add new code to decide
		what params to estimate and how that's done)
	- once this is done, create qualitative eval figures that show how trajectories are different
		under regularization vs no regularization. show this in the context of all eval metrics above.


3/27

- add functions in learning dynamics.py 
	- predict training loss
	- predict validation loss
- this should be integrated in inference.py optimization (e.g. printing these
		losses as we go)


	- cross-validation
		- leave out 10% random samples
			-keep latents for unobserved samples just ignore likelihood term for unobserved samples
			- compute posterior then predict samples
			- average across many different "random-folds".
		- predict next session. train a model using past data using elbo then predict next session. repeat for every session then average RMSE.


	- add cross-validation into current methods
		- randomly leave out some percentage of the data
			(dont compute likelihood for those data points)
		- after training, predict missing y's using map estimate of z.
			- alt: do MC approx of likelihood under a large number of samples from posterior. avg then if >.5 -> 1.
	- add dict that specifies which model params we will compute gradients
		w.r.t. use this dict to define optimized params as well.
	- add joint optimization of model params (here its just the scale)
		- show we can accurately predict this parameter.
		- do this well: e.g. use diff learning rate if nec. 




3/25
	what i've done:
		- train RNN on full set of features to predict y
		- used hidden representation h(x) instead of x showed elbo is lower on training data.
			- this doesn't make sense. in rnn case "data" x is high dimensional
				- which means we are not comparing two models under the same data.
					- compare after integrating out rnn parameters (see variationalRNN)
					- OR for now, write code to predict missing entries in data.
		- train new rnn on just x1, x2 not including other features
		- show concat this h'(x) with summary stats gives better performance. 

Gameplan:
	- add validation data to model (e.g. leave out 10% of data)
	- get VI working with original model.
		- have synthetic validation plots
			- show in simulation, we can properly recover all model parameters.
				- beta, lamda, alpha, etc.
	- extend alpha to a vector. does it allow for better predictions?
		- do variational bayes on model parameters. 
	- how to reason about relative magnitudes of loss vs regularization.
		- look into questions ryan and jonathan were asking about this.



	- figure out how to train rnn without overfitting
		- how to train rnn by leaving out some samples.
		- need validation data to measure performance.
		- use LSTM variation. 








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