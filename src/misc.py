# unvectorized log joint computation
		# unvectorized version
		logprob_2 = init_latent_logpdf.log_prob(latent_mean[0])
		# latents
		lx = []
		for i in range(1,T):
			transition_logpdf2 = Normal(latent_mean[i-1], self.transition_scale)
			lx.append(transition_logpdf2.log_prob(latent_mean[i]))
			logprob_2 += transition_logpdf2.log_prob(latent_mean[i])
		# observations
		yx = []
		for i in range(0,T):
			obs_logpdf2 = Normal(latent_mean[i], self.obs_scale)
			yx.append(obs_logpdf2.log_prob(x[i]))
			logprob_2 += obs_logpdf2.log_prob(x[i])

		embed()