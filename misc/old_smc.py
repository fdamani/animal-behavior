class SMC(object):
	'''
	sequential monte carlo
	try this on simple LDS model where log prior is (0,1)
	transition and obs_scale set to 0.1
	'''
	def __init__(self, model, 
				num_particles=100,
				init_prior = (0.0, 0.1),
				transition_scale = 0.1, T=50, isLearned=False):
		self.model = model
		self.q_init_latent_loc = torch.tensor([init_prior[0]], 
			requires_grad=isLearned, device=device)
		self.q_init_latent_log_scale = torch.tensor([math.log(init_prior[1])], 
			requires_grad=isLearned, device=device)
		self.q_transition_log_scale = torch.tensor([math.log(transition_scale)], 
			requires_grad=isLearned, device=device)

		self.num_particles = num_particles
		self.resampling_criteria = num_particles / 2.0
		self.T = T
		self.weights = torch.ones(self.num_particles, device=device)
		self.particles = torch.zeros((self.num_particles, self.T))
		self.inter_marginal = torch.zeros(self.T)

	def q_init_sample(self, num_samples=1):
		q_t = Normal(self.q_init_latent_loc, torch.exp(self.q_init_latent_log_scale))
		return q_t.sample((torch.Size((num_samples,)))) ## might want rsample() to get reparameterized gradients

	def q_sample(self, x, z_past):
		'''
			z_t ~ q_t(z_t | x_1:t, z_1:t-1)
		'''
		z_mean = z_past[-1]
		q_t = Normal(z_mean, torch.exp(self.q_transition_log_scale))
		return q_t.sample() ## might want rsample() to get reparamterized gradients()

	def q_init_logprob(self, z_sample):
		q_t = Normal(self.q_init_latent_loc, torch.exp(self.q_init_latent_log_scale))
		return q_t.log_prob(z_sample)

	def q_logprob(self, z_t, x, z_past):
		'''
			q_t(z_t | x_1:t, z_1:t-1)
		'''
		z_mean = z_past[-1]
		q_t = Normal(z_mean, torch.exp(self.q_transition_log_scale))
		return q_t.log_prob(z_t)

	def compute_incremental_weight(self, z_0_to_t, x_0_to_t):
		''' vectorize across particles first dim of z_0_to_t
			for particle i
			alpha_t (z_{1:t}^i) = p_t(x_t, z_t^i | x_{1:t-1}, z_{1:t-1}^i) / q_t(z_t^i | x_{1:t}, z_{1:t-1}^i)
	
			input: z_sample is z_t
			particle is z_1:t-1
			x is a vector x_{1:t}
		'''
		num = z_0_to_t.size(0)
		x_0_to_t = x_0_to_t.repeat(num, 1)
		log_wts = []
		for i in range(num):
			z_sample = z_0_to_t[i][-1] # confirm this is the right sample
			particle = z_0_to_t[i][:-1]
			# log p_t(x_t, z_t | x_1:t-1, z_1:t-1)
			log_p_x_z_t = self.model.logjoint_t(x_0_to_t[i], z_0_to_t[i])
			# log q_t(z_t | x_1:t, z_1:t-1 )
			if particle.nelement() == 0:
				log_q = self.q_init_logprob(z_sample)
			else:
				log_q = self.q_logprob(z_sample, x_0_to_t[i], particle)
			# log alpha = log p - log q
			log_weight = log_p_x_z_t - log_q
			log_wts.append(log_weight)
		return torch.cat(log_wts)

	def update_weights(self, alpha):
		'''
			multiply weights at time t-1 with incremental weights alpha for time t
		'''
		self.weights = self.weights * alpha

	def normalize_log_weights(self, log_weights):
		'''
			input: N particle log_weights
			return normalized exponentiated weights
		'''
		softmax = F.softmax
		norm_weights = softmax(log_weights, dim=0)
		return norm_weights

	def reset_weights(self):
		self.weights = torch.ones(self.num_particles, device=device)
		
	def multinomial_resampling(self):
		'''
			given logits or normalized weights
			sample from multinomial/categorical
			***make sure its resampling with replacement.

			*note we are sampling full particle trajectories
			x_1:t given weights at time point t.
		
			****ancestor sampling not reparameterizable. 
			****check if rsample() gives score function gradients
			****pass in argument grad = bool. to compare with and w/o score
				func estimator.

		'''
		# categorical over normalized weight vector for N particles
		# p is on simplex
		sampler = Categorical(self.weights)
		# sample indices over particles
		samples = sampler.sample(torch.Size((self.num_particles,)))
		# identify particles corresponding to indices
		self.particles = self.particles[samples]



	def compute_expected_value(self, weights, particles):
		return torch.sum(weights.unsqueeze(dim=-1) * particles, dim=0)
	def compute_variance(self, weights, particles):
		return torch.var(weights.unsqueeze(dim=-1) * particles, dim=0)

	def particle_filter(self, x):
		'''
			# time-step t=0: standard IS
				# sample z_0^i ~ q_0 for i = 1,...,N
				# compute normalized importance weights w_0^i, for i = 1,...,N
			# time-step t > 0
				# sample ancestor indices a_t-1^i ~ Cat(w_t-1)
				# update particles according to ancestor indices
				# sample z_t^i ~ q_t(z_t | ... )
				# append new z's to particle trajectories
				# compute new weights p/q and normalize.
		'''
		# time-step t=0: sample init particles and compute importance weights
		t=0
		# sample init particles
		init_samples = self.q_init_sample(self.num_particles)
		# append
		self.particles[:, 0] += init_samples.flatten()
		# compute weights
		init_log_wts = self.compute_incremental_weight(
				self.particles[:, 0:t+1], x[0:t+1])

		self.inter_marginal[t] = self.compute_inter_marginal_log_likelihood_vec(init_log_wts)
		#self.weights = self.normalize_log_weights(init_log_wts)

		logsoftmax = F.log_softmax
		self.weights = logsoftmax(init_log_wts, dim=0)

		# update log marginal likelihood
		# time-step t > 0
		for t in range(1, self.T):
			# sample ancestor indices, update particle trajectories
			#self.multinomial_resampling()
			# multinomial resampling
			sampler = Categorical(logits=self.weights)
			# sample indices over particles
			samples = sampler.sample(torch.Size((self.num_particles,)))
			# identify particles corresponding to indices
			self.particles = self.particles[samples]
			self.weights = torch.ones(self.num_particles, device=device)

			q_t = Normal(self.particles[:, t-1], torch.exp(self.q_transition_log_scale))
			self.particles[:, t] =  q_t.sample()
			self.weights = self.compute_incremental_weight(self.particles[:, 0:t+1], x[0:t+1])
			self.inter_marginal[t] = self.compute_inter_marginal_log_likelihood_vec(self.weights)
			self.weights = logsoftmax(self.weights, dim=0)
			print t

	def estimate(self, x):
		self.particle_filter(x)
		exp_value = self.compute_expected_value(torch.exp(self.weights), self.particles)
		var = self.compute_variance(torch.exp(self.weights), self.particles)
		return exp_value, var

	def compute_inter_marginal_log_likelihood_vec(self, log_wx):

		return (math.log(1.0) - math.log(self.num_particles) + torch.logsumexp(log_wx, dim=0)).unsqueeze(dim=0)

	def compute_inter_marginal_likelihood(self, t, wx):
		'''
			sum_t log 1/n sum_i w_t^i
			this an empirical approximation to the log marginal likelihood
			this is a function of self.weights
		'''
		self.inter_marginal[t] = torch.mean(wx)

	def compute_log_marginal_likelihood(self):
		'''
			sum over time
		'''
		return torch.sum(self.inter_marginal)

class IS(object):
	'''importance sampling
	'''
	def __init__(self, model, num_particles=10000, proposal_loc = 0.0, proposal_scale = 2.0):
		self.model = model
		self.num_particles = num_particles
		self.proposal_loc = torch.tensor(proposal_loc, requires_grad=False, device=device)
		self.proposal_scale = torch.tensor(proposal_scale, requires_grad=False, device=device)
		self.proposal_dist = Normal(self.proposal_loc, self.proposal_scale)

	def estimate(self, data):
		'''importance sample 
		'''
		# sample from proposal distribution
		samples = self.proposal_dist.sample(torch.Size((self.num_particles,1)))
		# compute weights
		log_weights = torch.zeros(self.num_particles, requires_grad=False, device=device)
		for i,sx in enumerate(samples):
			log_p_x_z = self.model.logjoint(data, sx)
			log_q = self.proposal_dist.log_prob(sx)
			log_wx = log_p_x_z - log_q
			log_weights[i] = log_wx
		# normalize weights
		softmax = F.softmax
		# probabilities on simplex
		norm_weights = softmax(log_weights)
		exp_value = self.compute_expected_value(norm_weights, samples)
		var = self.compute_variance(norm_weights, samples)
		return exp_value, var

	def normalize_weights(self, weights):
		return weights / torch.sum(weights)

	def compute_expected_value(self, weights, samples):
		return torch.dot(weights.flatten(), samples.flatten())

	def compute_variance(self, weights, samples):
		'''this might be the wrong variance
		E_q [w * f]
		variance of w*f evaluated under samples from q
		'''
		return torch.var(weights.flatten() * samples.flatten())