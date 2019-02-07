import numpy as np

class Config:
    title = 'no_title'
    label = 'no_label'
    iterations = 5

    display = True
    display_interval = 1000
    window_reward_training = 50000
    window_reward_test = window_reward_training
    window_loss = 5000
    window_q = 500

    # Agent
    max_step = 800000
    history_length = 4
    q_update_step = 4
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_end_step = 200000

    # Learner
    discount = 0.99
    learning_rate = 0.001
    target_q_update_step = 100

    # Memory
    memory_size = 50000
    learn_start_step = 100
    sample_interval = 4
    batch_size = 32
    sampling_method = 'random'
    imporatance_sampling = True
    last = True

    alpha = 0.6  # 0 corresponding to the uniform case
    beta = 0.4  # 1 fully compensates for the non-uniform probabilities
    beta_end_step = 200000
    minimum_td = 0.001

    replay_rate = 100
    replay_window = 30
    replay_start = 25

    # Environment
    laser_amount = 20
    laser_max_dist = 15.
    min_angle = -120.
    max_angle = 120.

    step_a_sec = 4.

    reward_goal = 1.
    reward_car = -1.
    reward_forward = 0.01
    reward_wall = 0.

    speed_ms = 0.83  # 1.4 m/s normal walking speed, 2.5 fast walking, 0.83 pepper
    turn_speed_rads = np.deg2rad(90)
    car_speed_ms = 1.4  # 8.33 m/s is 30 km/h, 4.17 is 15 km/h

    def setUp(self):
        self.speed = self.speed_ms / self.step_a_sec
        self.turn_speed = self.turn_speed_rads / self.step_a_sec
        self.car_speed = self.car_speed_ms / self.step_a_sec

        self.hidden_neurons = (self.history_length * self.laser_amount)-1

        length_map = 6.0
        self.max_reward_test = self.reward_goal / ((length_map/self.speed) + (np.deg2rad(180)/self.turn_speed))
        self.max_reward_training = (self.reward_goal + ((length_map / self.speed)*self.reward_forward)) / ((length_map / self.speed) + (np.deg2rad(180) / self.turn_speed))
        print("max reward test: ", self.max_reward_test)
        print("max reward training: ", self.max_reward_training)