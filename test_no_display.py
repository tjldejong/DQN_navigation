import threads_no_display
import pickle


class Tests:
    def test(self, configs):

        for config in configs:
            config.setUp()

            self.one_run(config, config.title + '_' + config.label)

    def one_run(self, config, title):
        # Need a lot of memory to save all these metrics, so only test agent performance is saved
        # steps_rewards, total_rewardss = [], []
        # steps_losss, total_losss = [], []
        # steps_qs, total_qs = [], []
        steps_rewards_test, total_rewardss_test = [], []

        for i in range(config.iterations):
            print(str(i + 1) + ' of ' + str(config.iterations) + ' in ' + title)
            steps_reward, total_rewards, steps_loss, losss, steps_q, qs, steps_reward_test, total_rewards_test = threads_no_display.one_run(
                config)

            # steps_rewards.append(steps_reward)
            # total_rewardss.append(total_rewards)
            # steps_losss.append(steps_loss)
            # total_losss.append(losss)
            # steps_qs.append(steps_q)
            # total_qs.append(qs)
            steps_rewards_test.append(steps_reward_test)
            total_rewardss_test.append(total_rewards_test)

        with open('data/' + title, 'wb') as data_file:
            # pickle.dump([steps_rewards, total_rewardss, steps_losss, total_losss, steps_qs, total_qs, steps_rewards_test, total_rewardss_test, config], data_file)
            pickle.dump([steps_rewards_test, total_rewardss_test, config], data_file)

