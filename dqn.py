import tensorflow as tf
import random as rand
import numpy as np
from convnet import ConvNet
from buff import Buffer
from memory import Memory


class DQN:

	def __init__(self, env, params):
		self.env = env
		params.actions = env.actions()
		self.num_actions = env.actions()
		self.episodes = params.episodes
        self.steps = params.steps
        self.history_length = params.history_length
        self.discount = params.discount
        self.eps = params.init_eps
        self.eps_delta = (params.init_eps - params.final_eps) / params.final_eps_frame
        self.replay_start_size = params.replay_start_size
        self.eps_endt = params.final_eps_frame
        self.batch_size = params.batch_size

        self.global_step = tf.Variable(0, trainable=False)
        if params.lr_anneal:
            self.lr = tf.train.exponential_decay(params.lr, self.global_step, params.lr_anneal, 0.96, staircase=True)
        else:
            self.lr = params.lr

        self.buffer = Buffer(params)
        self.memory = Memory(params.size, self.batch_size)

        with tf.variable_scope("train") as self.train_scope:
            self.train_net = ConvNet(params, trainable=True)
        with tf.variable_scope("target") as self.target_scope:
            self.target_net = ConvNet(params, trainable=False)

        self.optimizer = tf.train.RMSPropOptimizer(self.lr, params.decay_rate, 0.0, self.eps)

        self.actions = tf.placeholder(tf.float32, [None, self.num_actions])
        self.q = tf.placeholder(tf.float32, [None])
		self.rewards = tf.placeholder(tf.float32, [None])
		self.goals = tf.placeholder(tf.float32, [None])
        self.q_target = tf.add(self.rewards, tf.mul(1.0-self.goals, tf.mul(self.discount, self.q)))
        self.q_pred = tf.reduce_max(tf.mul(self.train_net.y, self.actions), reduction_indices=1)
        self.diff = tf.sub(self.q_target, self.q_pred)

		half = tf.constant(0.5)
		self.diff = tf.sub(self.q_target, self.q_pred)
		if params.clip_delta > 0:
			abs_diff = tf.abs(self.diff)
			clipped_diff = tf.clip_by_value(abs_diff, 0, 1)
    		linear_part = abs_diff - clipped_diff
    		quadratic_part = tf.square(clipped_diff)
    		self.diff_square = tf.mul(half, tf.add(quadratic_part, linear_part))
		else:
			self.diff_square = tf.mul(half, tf.square(self.diff))
        
        if params.accumulator == 'sum':
			self.loss = tf.reduce_sum(self.diff_square)
		else:
			self.loss = tf.reduce_mean(self.diff_square)

        self.task = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def randomRestart(self):
    	self.env.restart()
    	for _ in range(self.random_starts):
    		action = rand.randrange(self.num_actions)
    		reward = self.env.act(action)
      		state = self.env.getScreen()
      		terminal = self.env.isTerminal()
      		self.buffer.add(state)

    def trainEps(self, train_step):
    	if train_step < self.eps_endt:
      		return self.eps - train_step * self.eps_delta
    	else:
      		return self.eps_endt

    def observe(self, exploration_rate=self.eps):
    	if rand.random() < exploration_rate:
      		action = rand.randrange(self.num_actions)
    	else:
    		x = self.buffer.getAllStates()
    		action_values = self.train_net.y.eval(
            		feed_dict={ self.train_net.x: x })
            action = np.argmax(action_values)

    	reward = self.env.act(action)
    	state = self.env.getScreen()
    	terminal = self.env.isTerminal()
    	self.buffer.add(state)

      	return action, reward, state, terminal

    def doMinibatch(self, successes, failures):
        batch = self.memory.getSample()
        actions = np.array([batch[i][0] for i in range(self.batch_size)]).astype(np.float32)
        rewards = np.array([batch[i][1] for i in range(self.batch_size)]).astype(np.float32)
        successes += np.sum(rewards==1)
        failures += np.sum(rewards==-1)
        screens = np.array([batch[i][2] for i in range(self.batch_size)]).astype(np.float32)
        terminals = np.array([batch[i][3] for i in range(self.batch_size)]).astype(np.float32)

        q_target = self.target_net.y.eval( feed_dict={ self.target_net.x: screens } )
        q_target_max = np.argmax(q_target, axis=1)
        q_target = tf.add(rewards, tf.mul(1.0 - terminals, tf.mul(self.discount, q_target_max)))

        (result, loss) = self.sess.run( [self.task, self.loss],
                                        feed_dict={ self.q: q_target,
                                        			self.train_net.x: screens,
                                        			self.actions: actions } )

        return successes, failures, loss

    def play(self):
    	self.randomRestart()
    	for i in xrange(self.episodes):
      		terminal = False
      		while not terminal:
        		action, reward, screen, terminal = self.observe()

    def copy_weights(self, sess):
        for key in self.train_net.weights.keys():
            sess.run(self.target_net.weights[key].asssign(self.train_net.weights[key]))

    def save(self, saver, sess, step):
    	saver.save(sess, self.ckpt_dir, global_step=steps)

	def restore(self, saver):
		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)


