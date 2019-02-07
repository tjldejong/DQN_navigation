import displays
import pickle
import numpy as np

class PlotMaker:
    def __init__(self):
        pass

    def make_one_config_plot(self, title):
        # Load data from pickle file
        data = self.load_data(title)
        # Calculate moving averages
        steps_rewards, total_rewardss, total_rewards_mean, steps_losss, total_losss, losss_mean, steps_qs, total_qs, qs_mean, steps_rewards_test, total_rewardss_test, total_rewardss_test_mean = self.process_data(data)
        # Plot
        self.plot_data(steps_rewards, total_rewardss, total_rewards_mean, steps_losss, total_losss, losss_mean, steps_qs, total_qs, qs_mean, steps_rewards_test, total_rewardss_test, total_rewardss_test_mean, data[-1], title)

    def load_data(self, title):
        with open('data/' + title, 'rb') as data_file:
            data = pickle.load(data_file)
        return data

    def process_data(self, data):
        moving_average_reward, moving_average_loss, moving_average_q, moving_average_reward_test = [], [], [], []

        steps_rewards, total_rewardss, steps_losss, total_losss, steps_qs, total_qs, steps_rewards_test, total_rewardss_test, config = data

        # Because test run is asynchronous all test runs have a different amount of steps. Here all test are shortened to the shortest test.
        steps_rewards, total_rewardss = shorten_list(steps_rewards, total_rewardss)
        steps_losss, total_losss = shorten_list(steps_losss, total_losss)
        steps_qs, total_qs = shorten_list(steps_qs, total_qs)
        steps_rewards_test, total_rewardss_test = shorten_list(steps_rewards_test, total_rewardss_test)

        # Calculate moving average
        for i in range(config.iterations):
            moving_average_reward.append(np.asarray(moving_average(total_rewardss[i], config.window_reward_training))/config.max_reward_training)
            moving_average_loss.append(moving_average(total_losss[i], config.window_loss))
            moving_average_q.append(moving_average(total_qs[i], config.window_q))
            moving_average_reward_test.append(np.asarray(moving_average(total_rewardss_test[i], config.window_reward_test))/config.max_reward_test)

        # Calculate mean of moving averages
        total_rewards_mean = np.asarray(moving_average_reward).mean(axis=0)
        losss_mean = np.asarray(moving_average_loss).mean(axis=0)
        qs_mean = np.asarray(moving_average_q).mean(axis=0)
        total_rewardss_test_mean = np.asarray(moving_average_reward_test).mean(axis=0)

        # Normalize test steps, because of asynchronous more test steps are done than train steps
        steps_rewards_test[0] = np.divide(steps_rewards_test[0], np.max(steps_rewards_test[0]))

        return steps_rewards, moving_average_reward, total_rewards_mean, steps_losss, moving_average_loss, losss_mean, steps_qs, moving_average_q, qs_mean, steps_rewards_test, moving_average_reward_test, total_rewardss_test_mean

    def plot_data(self, steps_rewards, total_rewardss, total_rewards_mean, steps_losss, total_losss, losss_mean, steps_qs, total_qs, qs_mean, steps_rewards_test, total_rewardss_test, total_rewardss_test_mean, config, title):
        one_test_display = displays.OneTestDisplays(config)
        one_test_display.plot_one_test(steps_rewards, total_rewardss, total_rewards_mean, steps_losss, total_losss,
                                       losss_mean, steps_qs, total_qs, qs_mean, steps_rewards_test, total_rewardss_test,
                                       total_rewardss_test_mean, title)

    def make_multi_plot(self, configs):
        sr, mr, sl, ml, sq, mq, srt, mrt = [], [], [], [], [], [], [], []
        labels = []

        for config in configs:
            with open('data/'+config.title + '_' + config.label, 'rb') as data_file:
                data = pickle.load(data_file)

            steps_rewards, moving_average_reward, total_rewards_mean, steps_losss, moving_average_loss, losss_mean, steps_qs, moving_average_q, qs_mean, steps_rewards_test, moving_average_reward_test, total_rewardss_test_mean = self.process_data(data)

            sr.append(steps_rewards[0])
            mr.append(total_rewards_mean)
            sl.append(steps_losss[0])
            ml.append(losss_mean)
            sq.append(steps_qs[0])
            mq.append(qs_mean)
            srt.append(steps_rewards_test[0])
            mrt.append(total_rewardss_test_mean)
            labels.append(config.title + ' = ' + str(config.label))

        means_display = displays.MeanDisplays(configs[0])
        means_display.plot_means(sr, mr, sl, ml, sq, mq, srt, mrt, labels)


def shorten_list(list1, list2):
        min_size_list1 = min(map(len, list1))
        for i in range(len(list1)):
            list1[i] = list1[i][:min_size_list1]
            list2[i] = list2[i][:min_size_list1]
        return list1, list2


def moving_average(list, n):
    mov_avr = []
    cumsum = np.cumsum(list)
    for j in range(len(cumsum)):
        low = max(j-n,0)
        length = max(min(j,n),1)
        mov_avr.append(((cumsum[j]-cumsum[low])/length))
    return mov_avr