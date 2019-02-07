import tensorflow as tf
import numpy as np
import time


class Agent:
    def __init__(self, config, env, environment_queue, agent_performance_queue, memory_queue, name, environment_disp_queue = None):
        self.config = config

        self.environment = env
        self.environment_queue = environment_queue
        self.agent_performance_queue = agent_performance_queue
        self.memory_queue = memory_queue
        self.sess = tf.Session()
        self.name = name

        self.environment_disp_queue = environment_disp_queue

        self.training_steps, self.training_rewards = [], []

        self.state = np.zeros([config.history_length * config.laser_amount], dtype=np.float)

        self.build_dqn()

    def train(self, weights, lock_weights):
        step = -1
        ep = self.config.epsilon_start
        start_time = time.time()

        self.init_state()

        while True:
            step += 1

            # Action
            ep = max(ep - 1./self.config.epsilon_end_step, self.config.epsilon_end)
            lasers, action, reward, collision = self.action(ep)

            next_state = np.append(self.state[self.config.laser_amount:], lasers)

            # Learn
            if step % self.config.q_update_step == 0:
                self.learn_weights(weights, lock_weights)

            # Save
            self.training_steps.append(step)
            self.training_rewards.append(reward)
            if step % self.config.display_interval == 0:
                self.agent_performance_queue.put([self.training_steps, self.training_rewards])

            # Display
            if step % 4 == 0 and self.config.display:
                self.environment_queue.put(self.environment)

            # Store
            self.memory_queue.put([step, self.state, action, reward, next_state, collision, self.environment, True])

            if collision:
                self.new_game()
            else:
                self.state = next_state

            if step % 5001 == 0 and step != 0:
                end_time = time.time()
                print('step', step, 'of', self.config.max_step, 'ETF: {:.1f} min'.format((((end_time - start_time)/step) * (self.config.max_step - step))/60.))

    def action(self, ep):
        if np.random.random() < ep:
            action = np.random.randint(0, self.environment.action_amount)
        else:
            action = self.q_action.eval({self.s_t: [self.state]}, self.sess)[0]

        lasers, reward, collision = self.environment.act(action)

        return lasers, action, reward, collision

    def learn_weights(self, weights, weights_lock):
        # Obtain latest weights from learner (via Manager)
        locked = weights_lock.acquire(blocking=True)
        for name in weights.keys():
            self.w_assign_op[name].eval({self.w_input[name]: weights[name]}, session=self.sess)
        if locked:
            weights_lock.release()

    def init_state(self):
        # Performs 4 random actions to get an initial state
        for i in range(self.config.history_length):
            action = np.random.randint(0, self.environment.action_amount)
            self.state[(i*self.config.laser_amount):(i*self.config.laser_amount+self.config.laser_amount)], _, _ = self.environment.act(action)

    def new_game(self):
        self.environment.new_game()
        self.init_state()

    def test(self, weights, weights_lock):
        step = -1
        self.init_state()

        while True:
            step += 1

            # Action
            lasers, action, reward, collision = self.action(0.001)

            # Learn
            if step % self.config.q_update_step == 0:
                self.learn_weights(weights, weights_lock)

            if step % 3 == 0 and self.config.display and self.config.sampling_method != 'prioritized_mem':
                self.environment_disp_queue.put(self.environment)

            # Save
            self.training_steps.append(step)
            if abs(reward) < 0.9:
                reward = 0
            self.training_rewards.append(reward)

            if step % self.config.display_interval == 0:
                self.agent_performance_queue.put([self.training_steps, self.training_rewards])

            if collision or step % 1000 == 0:
                self.new_game()
            else:
                self.state = np.append(self.state[self.config.laser_amount:], lasers)

    def create_prioritized_memories(self, weights, lock_weights):
        step = -1

        while True:
            while self.environment_queue.qsize() > 0:
                self.state, self.environment = self.environment_queue.get()

                for i in range(self.config.replay_window):
                    step += 1

                    # Action
                    lasers, action, reward, collision = self.action(0.001)
                    next_state = np.append(self.state[self.config.laser_amount:], lasers)

                    # Learn
                    if step % self.config.q_update_step == 0:
                        self.learn_weights(weights, lock_weights)

                    # Display
                    if self.config.display and step % 4 == 0:
                        self.environment_disp_queue.put(self.environment)

                    self.memory_queue.put([step, self.state, action, reward, next_state, collision, self.environment, False])

                    if collision:
                        self.new_game()
                    else:
                        self.state = next_state

    def build_dqn(self):
        self.w = {}

        # Training network
        with tf.variable_scope(self.name):
            self.s_t = tf.placeholder('float32', [None, self.config.laser_amount * self.config.history_length], name='s_t')

            with tf.variable_scope('l1'):
                self.w['l1_w'] = tf.get_variable('Matrix',
                                                 [self.config.laser_amount * self.config.history_length, self.config.hidden_neurons],
                                                 tf.float32, tf.random_normal_initializer(stddev=0.02))
                self.w['l1_b'] = tf.get_variable('bias', [self.config.hidden_neurons],
                                                 initializer=tf.constant_initializer(0.0))

                self.l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.s_t, self.w['l1_w']), self.w['l1_b']))

            with tf.variable_scope('q'):
                self.w['q_w'] = tf.get_variable('Matrix', [self.config.hidden_neurons, self.environment.action_amount],
                                                tf.float32, tf.random_normal_initializer(stddev=0.02))
                self.w['q_b'] = tf.get_variable('bias', [self.environment.action_amount],
                                                initializer=tf.constant_initializer(0.0))

                self.q = tf.nn.bias_add(tf.matmul(self.l1, self.w['q_w']), self.w['q_b'])

            self.q_action = tf.argmax(self.q, axis=1)

        with tf.variable_scope('import_network'):
            self.w_input = {}
            self.w_assign_op = {}

            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float', self.w[name].get_shape().as_list(), name=name)
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

        tf.global_variables_initializer().run(session=self.sess)
