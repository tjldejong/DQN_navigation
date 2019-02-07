from config import Config
import pickle
import numpy as np
from matplotlib import pyplot as pp

window_reward_test = 50000


def moving_average(list, n):
    mov_avr = []
    cumsum = np.cumsum(list)
    for j in range(len(cumsum)):
        low = max(j-n, 0)
        length = max(min(j, n), 1)
        mov_avr.append(((cumsum[j]-cumsum[low])/length))
    return mov_avr


def shorten_list(list1, list2):
    min_size_list1 = min(map(len, list1))
    for i in range(len(list1)):
        list1[i] = list1[i][:min_size_list1]
        list2[i] = list2[i][:min_size_list1]
    return list1, list2


configs = [Config(),Config(),Config(),Config(),Config(),Config()]

configs[0].label = 'random'
configs[0].title = 'Sampling_method_r_normal'

configs[1].label = 'prioritized'
configs[1].title = 'Sampling_method_p_normal'

configs[2].label = 'age'
configs[2].title = 'Sampling_method_a_normal'

configs[3].label = 'sequences'
configs[3].title = 'Sampling_method_s_normal'

configs[4].label = 'hybrid'
configs[4].title = 'Sampling_method_h_normal'

configs[5].label = 'mem'
configs[5].title = 'Sampling_method_m_normal'

fig, ax = pp.subplots(figsize=(15, 15))
colors = list(pp.cm.tab10([0,1,2,3,4,5,6,7,8,9])) + ["crimson", "indigo", "lavender", "darkcyan", "cornflowerblue", "tomato"]

for i in range(len(configs)):
    title = configs[i].title
    with open('data/' + title, 'rb') as data_file:
        data = pickle.load(data_file)
    steps_rewards_test, total_rewardss_test, config = data

    steps_rewards_test, total_rewardss_test = shorten_list(steps_rewards_test, total_rewardss_test)

    mov_avr = []
    for j in range(len(steps_rewards_test)):
        mov_avr_plot = np.asarray(moving_average(total_rewardss_test[j], window_reward_test))/0.027
        mov_avr.append(mov_avr_plot)

    mean_mov_avr = np.mean(mov_avr, axis=0)

    ax.plot(steps_rewards_test[0] / np.max(steps_rewards_test[0]), mean_mov_avr, label=configs[i].label, color=colors[i])

    maxline = mean_mov_avr + np.std(mov_avr, axis=0)
    minline = mean_mov_avr - np.std(mov_avr, axis=0)

    ax.fill_between(steps_rewards_test[0] / np.max(steps_rewards_test[0]), minline, maxline, alpha=0.1, color=colors[i], linewidth=0)

ax.set_title('Test Reward')
ax.set_xlabel('Fraction of test')
ax.set_ylabel('Moving average ({}) reward'.format(window_reward_test))
ax.set_ylim(0, 0.8)
ax.set_xlim(0, 1.0)
ax.legend()

fig.savefig('figs/run50_normal_std.png')

pp.show()
