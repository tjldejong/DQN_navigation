import numpy as np
import tensorflow as tf


class Learner:
    def __init__(self, config, batch_queue, learner_performance_queue, update_p_queue):
        self.config = config
        self.sess = tf.Session()

        self.learn_steps, self.losss = [], []
        self.qs = []

        self.batch_queue = batch_queue
        self.learner_performance_queue = learner_performance_queue
        self.update_p_queue = update_p_queue

        self.build_dqn()

    def learn(self, weights, lock_weights):
        step = -1

        # Send weights to agent (via Manager)
        lock_weights.acquire()
        for name in self.w.keys():
            weights.update({name: self.w[name].eval(session=self.sess)})
        lock_weights.release()

        while True:
            step += 1

            batch = self.batch_queue.get()

            state_batch_all, action_batch_all, reward_batch_all, next_state_batch_all, collision_all, importance_sampling_weights_all, indexes = batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]

            state_batch = state_batch_all[0:self.config.batch_size]
            action_batch = action_batch_all[0:self.config.batch_size]
            reward_batch = reward_batch_all[0:self.config.batch_size]
            next_state_batch = next_state_batch_all[0:self.config.batch_size]
            collision = collision_all[0:self.config.batch_size]
            importance_sampling_weights = importance_sampling_weights_all[0:self.config.batch_size]

            # Double Q-learning
            pred_action = self.q_action.eval({self.s_t: next_state_batch}, session=self.sess)

            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                self.target_s_t: next_state_batch,
                self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
            }, session=self.sess)
            target_q_batch = (1. - collision) * self.config.discount * q_t_plus_1_with_pred_action + reward_batch

            # Optimize
            _, loss, q, delta = self.sess.run([self.optim, self.loss, self.q, self.delta], {self.target_q_t: target_q_batch, self.action: action_batch, self.s_t: state_batch, self.importance_sampling_weights: importance_sampling_weights})

            # Use double Q-learning to calculate new TD errors
            if self.config.sampling_method != 'random':
                pred_action = self.q_action.eval({self.s_t: next_state_batch_all}, session=self.sess)

                q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                    self.target_s_t: next_state_batch_all,
                    self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
                }, session=self.sess)
                target_q_batch_all = (1. - collision_all) * self.config.discount * q_t_plus_1_with_pred_action + reward_batch_all

                delta = self.delta.eval(feed_dict={self.target_q_t: target_q_batch_all, self.action: action_batch_all, self.s_t: state_batch_all},
                                session=self.sess)

                # Send TD errors
                self.update_p_queue.put([indexes, delta])
            else:
                self.update_p_queue.put([indexes, delta])

            # Save
            self.learn_steps.append(batch[0])
            self.losss.append(loss)

            self.qs.append(np.mean(q))

            # Display
            if step % self.config.display_interval == 0:
                self.learner_performance_queue.put([self.learn_steps,  self.losss, self.qs])

            # Send Q-network
            if step % self.config.q_update_step == 0:
                lock_weights.acquire()
                for name in self.w.keys():
                    weights.update({name: self.w[name].eval(session=self.sess)})
                lock_weights.release()

            # Copy weights and bias to target network
            if step % self.config.target_q_update_step == 0:
                for name in self.w.keys():
                    self.target_w_assign_op[name].eval({self.target_w_input[name]: self.w[name].eval(session=self.sess)}, session=self.sess)

    def build_dqn(self):
        self.w = {}
        self.target_w = {}

        action_amount = 3

        # Training network
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32', [None, self.config.laser_amount * self.config.history_length], name='s_t')

            with tf.variable_scope('l1'):
                self.w['l1_w']= tf.get_variable('Matrix', [self.config.laser_amount*self.config.history_length, self.config.hidden_neurons], tf.float32, tf.random_normal_initializer(stddev=0.02))
                self.w['l1_b'] = tf.get_variable('bias', [self.config.hidden_neurons], initializer=tf.constant_initializer(0.0))

                self.l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.s_t, self.w['l1_w']), self.w['l1_b']))

            with tf.variable_scope('q'):
                self.w['q_w']= tf.get_variable('Matrix', [self.config.hidden_neurons, action_amount], tf.float32, tf.random_normal_initializer(stddev=0.02))
                self.w['q_b'] = tf.get_variable('bias', [action_amount], initializer=tf.constant_initializer(0.0))

                self.q = tf.nn.bias_add(tf.matmul(self.l1, self.w['q_w']), self.w['q_b'])

            self.q_action = tf.argmax(self.q, axis=1)

        # Target network
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', [None, self.config.laser_amount * self.config.history_length], name='target_s_t')

            with tf.variable_scope('target_l1'):
                self.target_w['l1_w'] = tf.get_variable('Matrix',
                                                 [self.config.laser_amount * self.config.history_length, self.config.hidden_neurons], tf.float32, tf.random_normal_initializer(stddev=0.02))
                self.target_w['l1_b'] = tf.get_variable('bias', [self.config.hidden_neurons], initializer=tf.constant_initializer(0.0))

                self.target_l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.target_s_t, self.target_w['l1_w']), self.target_w['l1_b']))

            with tf.variable_scope('target_q'):
                self.target_w['q_w'] = tf.get_variable('Matrix', [self.config.hidden_neurons, action_amount], tf.float32, tf.random_normal_initializer(stddev=0.02))
                self.target_w['q_b'] = tf.get_variable('bias', [action_amount], initializer=tf.constant_initializer(0.0))

                self.target_q = tf.nn.bias_add(tf.matmul(self.target_l1, self.target_w['q_w']), self.target_w['q_b'])

            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
            self.target_w_input = {}
            self.target_w_assign_op = {}

            for name in self.w.keys():
                self.target_w_input[name] = tf.placeholder('float', self.target_w[name].get_shape().as_list(), name=name)
                self.target_w_assign_op[name] = self.target_w[name].assign(self.target_w_input[name])

        # Optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            self.importance_sampling_weights = tf.placeholder('float32', [None], name='importance_sampling_weights')

            action_one_hot = tf.one_hot(self.action, action_amount, on_value=1.0, off_value=0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            # TD error
            self.delta = tf.subtract(self.target_q_t, q_acted)

            # Huber loss with importance sampling weights
            self.loss = tf.losses.huber_loss(self.target_q_t, q_acted, weights=self.importance_sampling_weights)

            optimizer = tf.train.AdamOptimizer(self.config.learning_rate, epsilon=0.01)
            self.optim = optimizer.minimize(self.loss)

        tf.global_variables_initializer().run(session=self.sess)

        for name in self.w.keys():
            self.target_w_assign_op[name].eval({self.target_w_input[name]: self.w[name].eval(session=self.sess)}, session=self.sess)