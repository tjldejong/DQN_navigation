import numpy as np

class ReplayMemory:
    def __init__(self, config, memory_queue, batch_queue, update_p_queue, priority_environment_queue):
        self.config = config

        extra_mem = 100000

        self.states = np.empty((config.max_step+extra_mem, config.laser_amount * config.history_length), dtype=np.float)
        self.actions = np.empty(config.max_step+extra_mem, dtype=np.int)
        self.rewards = np.empty(config.max_step+extra_mem, dtype=np.float)
        self.next_states = np.empty((config.max_step+extra_mem, config.laser_amount * config.history_length), dtype=np.float)

        self.collisions = np.empty(config.max_step+extra_mem, dtype=np.bool)

        self.current = 0

        self.memory_queue = memory_queue
        self.batch_queue = batch_queue
        self.update_p_queue = update_p_queue
        self.priority_environment_queue = priority_environment_queue

        # PER parameters
        self.p = np.empty(config.max_step+extra_mem, dtype=np.float)
        self.alpha = config.alpha
        self.beta = config.beta
        self.eps = config.minimum_td

        self.last_updated = config.learn_start_step

        self.age = np.zeros(config.max_step+extra_mem, dtype=np.int)

        # Prioritized memories parameters
        self.replay_rate = config.replay_rate
        self.replay_start = config.replay_start
        self.env = []
        self.train_mem = np.zeros(config.max_step+extra_mem, dtype=np.bool)
        self.oldsum = 0

        if config.last:
            self.batch_length = config.batch_size - config.sample_interval
        else:
            self.batch_length = config.batch_size

        self.sequences_indexes = np.random.choice(self.config.learn_start_step, size=self.batch_length)
        self.sequences_count = 0
        self.importance_sampling_weight_sequences = np.ones(self.batch_length)

    def loop(self):
        step = -1
        beta_init = self.beta

        while True:
            while self.memory_queue.qsize() > 0:
                step += 1

                # Store
                mem = self.memory_queue.get()
                self.store(mem[1], mem[2], mem[3], mem[4], mem[5], mem[6], mem[7])

                # Update TD errors
                while self.update_p_queue.qsize() > 0:
                    TD_errors = self.update_p_queue.get()
                    self.update_p(TD_errors[0], TD_errors[1])

                # Sample
                if self.current > self.config.learn_start_step and step % self.config.sample_interval == 0:
                    self.sample(step)

                self.beta = min(self.beta + (1. - beta_init) / self.config.beta_end_step, 1.0)

    def store(self, state, action, reward, next_state, collision, env, train_mem):
        self.states[self.current] = state
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.next_states[self.current] = next_state

        self.collisions[self.current] = collision

        # Priority. 0.1 only used until learn_start_step. After that it is overwritten using the latest TD value.
        self.p[self.current] = 0.1

        # Only used in age
        self.age += 1
        self.age[self.current] = 1

        # Only used in prioritized_mem, Save copy of environment and keep track of train agents memories
        self.env.append(env)
        self.train_mem[self.current] = train_mem

        self.current += 1

    def update_p(self, indexes, delta):
        self.p[indexes] = np.power((np.abs(delta) + self.eps), self.alpha)
        self.last_updated = np.max(indexes)
        assert len(indexes) == self.batch_length + self.config.sample_interval

    def sample(self, step):
        # Window of indexes to pick from.
        possible_indexes = np.arange(max(0, self.last_updated - self.config.memory_size), self.last_updated)

        # Pick random indexes
        if self.config.sampling_method == 'random':
            chosen_indexes = np.random.choice(len(possible_indexes), size=self.batch_length)
            importance_sampling_weight = np.ones(len(chosen_indexes))

        # Pick prioritized indexes
        if self.config.sampling_method == 'prioritized':
            P = self.p[possible_indexes]
            P = P / np.linalg.norm(P, 1)  # np.linalg.norm = max(sum(abs(x), axis=0))

            chosen_indexes = np.random.choice(len(possible_indexes), size=self.batch_length, p=P, replace=True)

            if self.config.imporatance_sampling:
                importance_sampling_weight = np.power((len(possible_indexes) * P[chosen_indexes]), -self.beta)
            else:
                importance_sampling_weight = np.ones(len(chosen_indexes))

        # Pick indexes using prioritized with an age factor
        if self.config.sampling_method == 'age':
            agefactor = np.power(self.age[possible_indexes], -0.05)

            p_age = agefactor * self.p[possible_indexes]
            P = p_age / np.linalg.norm(p_age, 1)

            chosen_indexes = np.random.choice(len(possible_indexes), size=self.batch_length, p=P, replace=True)

            if self.config.imporatance_sampling:
                importance_sampling_weight = np.power((len(possible_indexes) * P[chosen_indexes]), -self.beta)
            else:
                importance_sampling_weight = np.ones(len(chosen_indexes))

        # Pick half of indexes using prioritized and pick the indexes prior to those
        if self.config.sampling_method == 'sequences':
            P = self.p[possible_indexes]
            P = P / np.linalg.norm(P, 1)

            chosen_indexes_prioritized = np.random.choice(len(P), size=int(self.batch_length/2), p=P, replace=True)
            chosen_indexes = np.append(chosen_indexes_prioritized, chosen_indexes_prioritized - 1)

            if self.config.imporatance_sampling:
                importance_sampling_weight_chosen = np.power((len(possible_indexes) * P[chosen_indexes_prioritized]), -self.beta)
            else:
                importance_sampling_weight_chosen = np.ones(len(chosen_indexes_prioritized))

            importance_sampling_weight = np.append(importance_sampling_weight_chosen, importance_sampling_weight_chosen)

        # Pick half indexes random and half prioritized
        if self.config.sampling_method == 'hybrid':
            P = self.p[possible_indexes]
            P = P / np.linalg.norm(P, 1)

            chosen_indexes_prioritized = np.random.choice(len(possible_indexes), size=int(self.batch_length/2), p=P, replace=True)
            chosen_indexes_random = np.random.randint(len(possible_indexes), size=int(self.batch_length/2))
            chosen_indexes = np.append(chosen_indexes_prioritized, chosen_indexes_random)

            if self.config.imporatance_sampling:
                importance_sampling_weight_prioritized = np.power((len(possible_indexes) * P[chosen_indexes_prioritized]), -self.beta)
                importance_sampling_weight_random = np.ones(len(chosen_indexes_random))

                importance_sampling_weight = np.append(importance_sampling_weight_prioritized, importance_sampling_weight_random)
            else:
                importance_sampling_weight = np.ones(len(chosen_indexes))

        # Pick indexes prioritized
        if self.config.sampling_method == 'prioritized_mem':
            P = self.p[possible_indexes]
            P = P / np.linalg.norm(P, 1)

            chosen_indexes = np.random.choice(len(possible_indexes), size=self.batch_length, p=P)

            importance_sampling_weight = np.power((len(possible_indexes) * P[chosen_indexes]), -self.beta)

            # Periodically pick a state and environment to replay. Pick from the training agents experiences using the prioritized approach.
            if np.sum(self.train_mem) >= (self.oldsum + self.replay_rate):
                total_indexes = np.arange(max(self.last_updated - self.config.memory_size, self.replay_start), self.last_updated)
                train_indexes = total_indexes[self.train_mem[max(self.last_updated - self.config.memory_size, self.replay_start):self.last_updated]]

                P = self.p[train_indexes]
                P = P / np.linalg.norm(P, 1)

                chosen_p = np.random.choice(len(train_indexes), size=1, p=P, replace=True)
                ind_max_p = chosen_p[0]

                states_train = self.states[train_indexes]
                env_train = list(self.env[i] for i in train_indexes)

                state = states_train[ind_max_p - self.replay_start]
                env = env_train[int(ind_max_p) - self.replay_start]

                self.priority_environment_queue.put([state, env])

                self.oldsum = np.sum(self.train_mem)

        # Convert chosen indexes to indexes in memory
        indexes = possible_indexes[chosen_indexes]

        # Add newest experiences to batch. Depending on config.last these are used in the learner.
        for i in range(self.config.sample_interval):
            indexes = np.append(indexes, self.current - 1 - i)

        # The newest experiences get the average importance sampling weight
        importance_sampling_weight = np.append(importance_sampling_weight, np.mean(importance_sampling_weight) * np.ones(self.config.sample_interval))

        states = self.states[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        collision = self.collisions[indexes]
        next_states = self.next_states[indexes]

        self.batch_queue.put([step, states, actions, rewards, next_states, collision, importance_sampling_weight, indexes])

