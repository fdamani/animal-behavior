5/2

- add prior on scale
	- try map estimation using adam vs sgd vs lbfgs
	- try variational posterior



4/30:
to do now:
- implement residual block bootstrap estimation procedure. use this to estimate
	multiple gamma. do estimates look better? X

- re-run scripts
	- multiple alpha no gamma
	- multiple gamma no alpha
	- multiple gamma, multiple alpha


- running now:
	- multiple alpha no gamma
	- multiple gamma single alpha.
	- 

- validation
	- held out rat
		- sample from prior and evaluate likelihood. 


- validation
	- held out rat.
		- write separate script that takes in all rats. computes log prob of each of them
		then sums them and computes gradients. 
		- for now figure out how to compute a single theta estimate.
			- this involves log prob summing over rats and finding single estimates of the gradient.
			- in this view you can treat rats as iid and bootstrap over rats. 
			- and bootstrap over blocks.
	- held out time points
		- sample 



- write boxplot_point_estimate_across_rats X
	- for each alpha dimension compute boxplot across rats. 
		- are there interesting differences in these estimates across features?
- currently running multiple_gamma_multiple_alpha_old_parameterization
- waiting on uncertainty estimates for multiple_alpha_old_parameterization
	- currently have a plot of boxplot for each feature across rats using MLE estimates.
- 



- results from multiple gamma, single alpha, old parameterization makes sense
	4.29/multiple_gamma_single_alpha_old_parameterization
- results from multiple alpha, single gamma did not look good
	- running again with old parameterization and single gamma = 1e-20. (fixed)
		/multiple_alpha_old_parameterizatoin
- need to implement block residual bootstrap
	- re-estimate multiple alphas. are these estimates well calibrated?

- figure out how to do validation
	- for held out session
		- simulate data? 
		- use smc? 
- across rats take opt params and compute confidence interval
	of initial point estimates across rats.

4/29
jobs running tonight:
- multiple alpha.
- multiple gamma and single alpha (with new parameterization)



we want to run:
- multiple gamma, single alpha old parameterization.
	- we have these results for a diff # of obs/time point.
- multiple gamma, multiple alpha old and new paramterization. 







- single alpha
- multiple alpha
- multiple gamma single alpha
- multiple gamma multiple alpha
- multiple gamma multiple alpha with separate params for each term



in order:
1. implement cross-validation using 2nd order method X
2. print additional plot for held out session latent trajectory X
	- not obvious how to do held out session considering prior is recurrent.
		- simulate data and compute prior under it.
	- for diffusion prior it learns a horizontal line this makes sense because that
		is most likely under just the prior?
3. change alpha to just be alpha * learning 
4. compare single alpha model to vector of alphas across rats.
5. add feature-specific regularization. how does it do?
6. ideally this is under many different held out sessions. (beginning, middle, end)

- implement quick cross-validation using 2nd order method and test marginal likelihood using mean
- need some plots to illustrate why its best to fit scale with cross-validation
	- optimizing scale results in high-variance low bias fit
		- plot of train vs test accuracy across rats under this scale param.
		- marginal likelihood under the two models. 




- implement leave a session out (300 trials). (for now leave out one of the middle sessions)
	- two plots. latent trajectories with vertical lines around test sesssion
	- separate plot of test session
- have a plot comparing scale performance via c.v. and via elbo optimization
	- for c.v. 
		- fine to use map estimate from 2nd order method and evaluate posterior predictive
		under this quantity
	- compare this to elbo optimization.
- implement feature-specific learning rates with no regularization
- implement held out rat
	- product over rats
	- sample from prior and evaluate likelihood under prior samples

4/14
- train on last 5k trials and first 5k trials
	- bootstrap estimate for 6 params
	- repeat across datasets


4/13
	- learn 4 parameters
		- sparsity for x1 x2
		- sparsity for everything else
		- first half of trials versus second half.

	- try having a sparsity parameter per dimension
		- useful parameter that tells you something interesting about how the rat "pruns" out
			irrelevant dimensions.
			- try this in particular towards the latter half. 

	- finish evaluation plots
		- create plots of average and each mc trajectory for future
		- run results for 10 rats with 10 bootstrap estimates to get confidence intervals.
		- save eval results
		- run results for each rat broken into first half vs second half for 10 rats and 10 bootstraps per.




4/12

- investigate the claim "regularization improves future marginal lh"
	- we have reproduced this effect for varying number of future time steps and num obs/time-step for training
	- can we show this effect for one obs per time-step?
	- what about across rats?
	- can you visualize the trajectories and look at how they differ?
		- why does regularization give us more realistic trajectories? look at a few samples and maybe average
			over a bunch of them and compare 
- change z trajectory plots so they all have the same y axis (-4 to 4 is fine and make them wider/longer)




- get confidence intervals for two params and repeat this for 10 datasets.
	- run this on server, but get this working locally first.
- show we get the same estimates when estimating parameters on first half
	vs second half
- for 10 datasets, use the same # of trials (cut off trials when learning
	plateaus))

- get all other evaluation metrics to work first.


4/11

- automate all evaluation figures and have those figures produced every run
	- BIC values (automatically computed by number of params that are being optimized).
	- posterior predictive
	- marginal likelihoods
	- held-out-session trajectories predictions. 
	- PPCs.
	- avg likelihood vs plot of marginal likelihood as a function of data point.
		- use this to visualize where sparsity param might be helping?
		- plot max and mi likelihood across time points
			- does sparsity help us be more confident in what we know most well
			or does it help in the places that we were initially least confident?
			

we want to understand the effect of sparsity
	- X print relative contributions of learning and regularization component
	- repeat analysis under different scales
	- different number of trials
	- different type of regularization



- print magnitude of grad_rat_obj vs regularization for trained model
- repeat with smaller num trials
- re-do analysis for L1.



	- get bootstrap confidence interval plots for *just* l2 penalty. 
		- quick n dirty with big lr is fine for now.
	- get this working for one real dataset.
		- compare to gamma set to 0.
	- repeat for actual dataset as well. 
	- compare actual trajectories - how do they look different?



	- we have uncertainty plots for parameters
	- now, lets investigate the effect of regularization
		- confounding variables: scale size and number of trials per step.
		- run a few more runs with smaller # of trials per step. try 10 and 25.
		- plot magnitude of sparsity component vs learning component over time.


4/10
	- give me boxplots for bootstrapped results.
	- learning rate decay schedule for quick learning.
	- 




	- add adaptive learning rate that decays over time 
	- save all model parameters to folders
	- once we get results from synthetic datasets showing we can estimate model parameters accurately
		run on real datasets
	- implement cross-validation of scale parameter (decide based on map estimate?)
	- given data generated from this system, show loss is lower for this model versus one without regularization and no learning at all.
		- generate data according to this system and compare loss to a model where regularization 
			param is set to 0.
	- evaluation: rekindle prediction on validation data (all sorts of validation data)
		- predict these numbers for all models.
	- alpha
		- vector of alphas
		- decaying alpha
	- add momentum term
	- beta
		- why is it hard to estimate when beta=5?
		- can we estimate gamma better when beta=0.0 (e.g. l1 penatly?)








	- just finished draft of bootstrap code.
	- fix bugs, get plots working (confidence intervals for each parameter)
		- also make sure loss functions are properly saved for initial fit (its fine for now not to plot
			each bootsrap run. add boolean flag to inference class to not plot those.)
	- investigate different betas and confidence intervals.



	- run on real datasets.






	- we can properly estimate beta when its close to 0 or 1, but our single experiment for beta=.5 shows
		estimation is unreliable.
	- we should repeat these 3 experiments under a bootstrap scenario to get error bars on these estimates.
	- get bootstrap code working. compute error bars for each model parameter try many values of beta
		(0.0, 0.25, 0.5, 0.75, 1.0)


	- for now, lets not learn the scale parameter. set it to a reasonable value (we will cross-validate
		this value w.r.t. ELBO)
			- mean-field implies underestimation of posterior correlation implying high scale parameter
				b/c each q(z_i) wants to be independent so not much smoothness. todo list includes
				learn a more structured variational family but in the meantime i propose we be practical:
				cross-validate an estimate for the scale and move forward with interesting science.
	- show we can reliably estimate alpha, sparsity for l2 and now lets introduce second layer of sparsity
		show we can recover beta as well.
	- once we can successfully do above, lets get a boostrap estimate of all of these in synthetic data
		create plot and save for meeting with ryan tomorrow.
	- repeat with real data.
		- provide additional characterizations of sparsity. look at trajectories how do they vary
		- look at held out predictions on future trajectories.




	- show reliable estimation of sparsity parameter for l2
		- try separating learning and regularization parameter
			- instead of having alpha * loss
		- try different parameter settings -> which setting makes it most obviously identifiable?
		- try estimating sparsity without the learning component. is that identifiable?
	- estimate all params from synthetic data (show recovery)
	- estimate all params from real data (show confidence intervals)

4/9
	- we have sgd working with lr = 1e-1 (rough approximations but different estimates
		track expected behavior)
	- bootstrap estimation
		- fit a model to data
		- generate bootstrapped datasets according to theta hat
		- estimate theta_1:B
		- compute mean and variance of theta_1:B.

	- model building in simulation
		- add learning component and regularization
		- extend to vector of alphas
	- fit all these models to real data.


	- added l2 penalty
		- identify cases where this is an identifiable parameter. how does it tradeoff with other params?





4/8

- quick and dirty experiments for now.
- high learning rate fast convergence (albeit bouncy)
- implement bootstrap approach get quick results quick confdience intervals

- repeat with new model params.

- two experiments:
	- for single theta, generate 20 trajectory y's
		- fit the elbo to each of the 20
		- plot the estimated sigma's across 20 trajectories. 
	- bootstrap
		- generate data. fit one model
		- given theta hat, generate 20 bootstrapped datasets
		- fit bootstrapped datasets
		- estimate variance of theta hats from 1:B
		- bias correction. 


- try same scale for gradients of model parameters and gradients of vi params
	- whichever one is smaller multiple by a param to make it bigger.
- try sgd with momentum.
- how to get a better map fit?


	- what is a data regime where we are confident that it should work?
		- lots of time points and observations.
	- if i initialize phi_mu to true then just optimize theta
		- theta goes to the correct answer. this finding is robust to
		theta initialization (several orders of magnitude bigger or smaller
		it doesnt matter converges very quickly)
		- should i just find a really good map estimate?


	- map is too much because prior is so strong.

	- if we condition on true z and reasonable scale, can we 
		optimize theta?



	- goal: show that we can accurately estimate theta (at convergence)
		- try alternating optimization of phi and theta
		- different optimizers--adagrad, rmsprop.
	- bootstrap estimates (create plots just for sigma--show confidence intervals as
		a function of data pts)

4/7
	- implement bootstrap approach
		- sgd converges to correct mle estimate immediately then drifts away as loss goes down.
			- figure out what part of loss is making it go down. **entropy is part of it. 
				- anneal entropy? too strong?
			- after 10k iters and theta is at 0.046, try reducing uncertainty of variational
				params and reevaluate loss. where does it go? look at the two terms in inference
				see how they trade off.
		- do we notice same issues with other params like alpha?








	X - initialize VI with map estimate. it HELPS!
	X - then do joint optimization of vi and model params using SGD (try with momentum)
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
