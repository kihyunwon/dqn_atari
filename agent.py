import tensorflow as tf
from memory import Memory
from dqn import DQN

class Agent:

    def __init__(self, actions, params):
    	params.actions = actions
		self.params = params
		self.sess = tf.Session()
		self.mem = Memory(self.params)
		self.create_networks()

	def create_networks(self):
		print 'Creating Q network and its target network'
		self.q_net = DQN(self.params)
		self.target_net = DQN(self.params)
		self.sess.run(tf.initialize_all_variables())
		save_dict = { 'q_w1': self.q_net.conv1_w,'q_b1':self.q_net.conv1_b,
					  'q_w2': self.q_net.conv2_w,'q_b2':self.q_net.conv2_b,
					  'q_w3': self.q_net.conv3_w,'q_b3':self.q_net.conv3_b,
					  'q_w4': self.q_net.fc1_w,'q_b4':self.q_net.fc1_b,
					  'q_w5': self.q_net.fc2_w,'q_b5':self.q_net.fc2_b,
					  't_w1': self.target_net.conv1_w,'t_b1':self.target_net.conv1_b,
					  't_w2': self.target_net.conv2_w,'t_b2':self.target_net.conv2_b,
					  't_w3': self.target_net.conv3_w,'t_b3':self.target_net.conv3_b,
					  't_w4': self.target_net.fc1_w,'t_b4':self.target_net.fc1_b,
					  't_w5': self.target_net.fc2_w,'t_b5':self.target_net.fc2_b,
					  'step': self.q_net.global_step }
		self.saver = tf.train.Saver(save_dict)

		self.copy = [ self.target_net.conv1_w.assign(self.q_net.conv1_w),
					  self.target_net.conv1_b.assign(self.q_net.conv1_b),
					  self.target_net.conv2_w.assign(self.q_net.conv2_w),
					  self.target_net.conv2_b.assign(self.q_net.conv2_b),
					  self.target_net.conv3_w.assign(self.q_net.conv3_w),
					  self.target_net.conv3_b.assign(self.q_net.conv3_b),
					  self.target_net.fc1_w.assign(self.q_net.fc1_w),
					  self.target_net.fc1_b.assign(self.q_net.fc1_b),
					  self.target_net.fc2_w.assign(self.q_net.fc2_w),
					  self.target_net.fc2_b.assign(self.q_net.fc2_b) ]
		
		self.sess.run(self.copy)
		
		if self.params.ckpt_file is not None:
			print 'Loading checkpoint file: %s' % self.params.ckpt_file
			self.saver.restore(self.sess, self.params.ckpt_file)
			prev_step = self.sess.run(self.q_net.global_step)
			print 'Continue from %dth step' % prev_step

	def restartRandom(self):
    	self.env.restart()
    	# perform random number of dummy actions to produce more stochastic games
    	for i in xrange(random.randint(self.history_length, self.random_starts) + 1):
      		reward = self.env.act(0)
      		screen = self.env.getScreen()
      		terminal = self.env.isTerminal()
      		assert not terminal, "terminal state occurred during random initialization"
      		# add dummy states to buffer
      		self.buf.add(screen)

    def step(self, exploration_rate):
    	# exploration rate determines the probability of random moves
    	if random.random() < exploration_rate:
      		action = random.randrange(self.num_actions)
      		logger.debug("Random action = %d" % action)
    	else:
      		# otherwise choose action with highest Q-value
      		state = self.buf.getStateMinibatch()
      		# for convenience getStateMinibatch() returns minibatch
      		# where first item is the current state
      		qvalues = self.net.predict(state)
      		assert len(qvalues[0]) == self.num_actions
      		# choose highest Q-value of first state
      action = np.argmax(qvalues[0])
      logger.debug("Predicted action = %d" % action)

    # perform the action
    reward = self.env.act(action)
    screen = self.env.getScreen()
    terminal = self.env.isTerminal()

    # print reward
    if reward <> 0:
      logger.debug("Reward: %d" % reward)

    # add screen to buffer
    self.buf.add(screen)

    # restart the game if over
    if terminal:
      logger.debug("Terminal state, restarting")
      self._restartRandom()

    # call callback to record statistics
    if self.callback:
      self.callback.on_step(action, reward, terminal, screen, exploration_rate)

    return action, reward, screen, terminal

     def train(self, train_steps, epoch = 0):
    # do not do restart here, continue from testing
    #self._restartRandom()
    # play given number of steps
    for i in xrange(train_steps):
      # perform game step
      action, reward, screen, terminal = self.step(self._explorationRate())
      self.mem.add(action, reward, screen, terminal)
      # train after every train_frequency steps
      if self.mem.count > self.mem.batch_size and i % self.train_frequency == 0:
        # train for train_repeat times
        for j in xrange(self.train_repeat):
          # sample minibatch
          minibatch = self.mem.getMinibatch()
          # train the network
          self.net.train(minibatch, epoch)
      # increase number of training steps for epsilon decay
      self.total_train_steps += 1

	def play(self):
    	self.restartRandom()
    	for i in xrange(self.params.episodes):
      		done = False
      		while not done:
        		action, reward, screen, done = self.step(self.exploration_rate_test)
