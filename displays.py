import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as pp
import matplotlib.patches as patches

def close_all():
    pp.close('all')


# Live display of the road environment
class GameDisplay:
    def __init__(self, road_environment):
        self.fig, self.ax = pp.subplots()

        pp.get_current_fig_manager().window.setGeometry(0, 20, 550 * (road_environment.map.width / 20),
                                                        550 * (road_environment.map.length / 20))
        self.ax.set_xlim(0, road_environment.map.width)
        self.ax.set_ylim(0, road_environment.map.length)

        self.line, = self.ax.plot([road_environment.robot.x], [road_environment.robot.y], 'bo')
        self.line2, = self.ax.plot(
            [road_environment.robot.x, (road_environment.robot.x + 1 * np.sin(road_environment.robot.theta))],
            [road_environment.robot.y, (road_environment.robot.y + 1 * np.cos(road_environment.robot.theta))])

        # laserx, lasery, _ = road_environment.robot.get_laser_distance(road_environment.cars)
        # self.line3, = self.ax.plot(laserx, lasery, 'ro')
        self.line4, = self.ax.plot([], [])

        self.squares = []
        for car in road_environment.cars:
            self.squares.append(patches.Rectangle((car.x - (0.5 * car.length), car.y - (0.5 * car.width)), car.length, car.width, ))

        self.square_bottom = patches.Rectangle((0, 0), road_environment.map.width, road_environment.map.sidewalkwidth,
                                          facecolor="red")
        self.square_top = patches.Rectangle((0, road_environment.map.length - road_environment.map.sidewalkwidth),
                                       road_environment.map.width, road_environment.map.sidewalkwidth,
                                       facecolor="green")

        for square in self.squares:
            self.ax.add_patch(square)
        self.ax.add_patch(self.square_bottom)
        self.ax.add_patch(self.square_top)
        pp.show(block=False)
        pp.pause(0.01)

    def render(self, road_environment):
        self.line.set_data(road_environment.robot.x, road_environment.robot.y)
        self.line2.set_data(
            [road_environment.robot.x, (road_environment.robot.x + 1 * np.sin(road_environment.robot.theta))],
            [road_environment.robot.y, (road_environment.robot.y + 1 * np.cos(road_environment.robot.theta))])
        # laserx, lasery, _ = road_environment.robot.get_laser_distance(road_environment.cars)
        # self.line3.set_data(laserx, lasery)
        x, y = self.calc_lasers(road_environment)
        self.line4.set_data(x, y)
        for i, square in enumerate(self.squares):
            square.set_x(road_environment.cars[i].x - (0.5 * road_environment.cars[i].length))

        if road_environment.up:
            self.square_bottom.set_facecolor("red")
            self.square_top.set_facecolor("green")
        else:
            self.square_bottom.set_facecolor("green")
            self.square_top.set_facecolor("red")

        self.ax.draw_artist(self.ax.patch)
        for square in self.squares:
            self.ax.draw_artist(square)
        self.ax.draw_artist(self.square_bottom)
        self.ax.draw_artist(self.square_top)
        self.ax.draw_artist(self.line4)
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.line2)
        # self.ax.draw_artist(self.line3)

        self.fig.canvas.update()
        self.fig.canvas.flush_events()

    def calc_lasers(self, road_env):
        theta = road_env.robot.theta
        min_laser_angle = road_env.robot.min_angle_laser
        max_laser_angle = road_env.robot.max_angle_laser
        amount_lasers = road_env.robot.lasers
        thetas = np.linspace(theta + np.deg2rad(min_laser_angle),
                             theta + np.deg2rad(max_laser_angle), amount_lasers)

        x = np.empty(0)
        y = np.empty(0)

        _, _, r = road_env.robot.get_laser_distance(road_env.cars)

        i = 0
        for theta in thetas:
            x = np.append(x, [road_env.robot.x, road_env.robot.x+r[i]*np.sin(theta)])
            y = np.append(y, [road_env.robot.y, road_env.robot.y+r[i]*np.cos(theta)])
            i += 1

        return x, y


# Live displays of the performances of the train agent
class AgentDisplays:
    def __init__(self, config):
        self.config = config

        self.fig_train, self.ax_train = pp.subplots()
        pp.get_current_fig_manager().window.setGeometry(1920 - 550, 580, 550, 550)
        self.ax_train.set_xlim(0, config.max_step)
        self.ax_train.set_ylim(0, 1)
        self.line_train, = self.ax_train.plot([None], [None])
        pp.ylabel('Moving average ({}) reward'.format(config.window_reward_training))
        pp.xlabel('step')
        pp.show(block=False)
        pp.pause(0.1)

    def render(self, step, reward):
        moving_avr = np.divide(moving_average(reward, self.config.window_reward_training), self.config.max_reward_training)
        # moving_avr = np.multiply(moving_average(reward, self.config.window_reward_training), 1000)

        self.line_train.set_data(step, moving_avr)
        self.ax_train.draw_artist(self.ax_train.patch)
        self.ax_train.draw_artist(self.line_train)
        self.fig_train.canvas.update()
        self.fig_train.canvas.flush_events()


# Live displays of the performances of the learner
class LearnerDisplays:
    def __init__(self, config):
        self.config = config

        self.fig_loss, self.ax_loss = pp.subplots()
        pp.get_current_fig_manager().window.setGeometry(570, 570, 550, 550)
        self.ax_loss.set_xlim(0, config.max_step)
        self.ax_loss.set_ylim(0, 0.01)
        self.line_loss, = self.ax_loss.plot([None], [None])
        pp.ylabel('Moving average ({}) loss'.format(config.window_loss))
        pp.xlabel('steps')
        pp.show(block=False)
        pp.pause(0.1)

        self.fig_q, self.ax_q = pp.subplots()
        pp.get_current_fig_manager().window.setGeometry(0, 550, 550, 550)
        self.ax_q.set_xlim(0, config.max_step)
        self.ax_q.set_ylim(0, 4.0)
        self.line_q, = self.ax_q.plot([None], [None])
        pp.ylabel('Moving average ({}) mean q acted'.format(config.window_q))
        pp.xlabel('steps')
        pp.show(block=False)
        pp.pause(0.1)

    def render(self, step, loss, q):
        moving_avr_loss = moving_average(loss, self.config.window_loss)
        moving_avr_q = moving_average(q, self.config.window_q)

        self.line_loss.set_data(step, moving_avr_loss)
        self.ax_loss.draw_artist(self.ax_loss.patch)
        self.ax_loss.draw_artist(self.line_loss)
        self.fig_loss.canvas.update()
        self.fig_loss.canvas.flush_events()

        self.line_q.set_data(step, moving_avr_q)
        self.ax_q.draw_artist(self.ax_q.patch)
        self.ax_q.draw_artist(self.line_q)
        self.fig_q.canvas.update()
        self.fig_q.canvas.flush_events()


# Live displays of the performances of the test agent
class TestDisplays:
    def __init__(self, config):
        self.config = config

        self.fig_test, self.ax_test = pp.subplots()
        pp.get_current_fig_manager().window.setGeometry(1920-550, 20, 550, 550)
        self.ax_test.set_xlim(0, config.max_step)
        self.ax_test.set_ylim(0, 1)
        self.line_test, = self.ax_test.plot([None], [None])
        pp.ylabel('Moving average ({}) reward'.format(config.window_reward_training))
        pp.xlabel('step')
        pp.show(block=False)
        pp.pause(0.1)

    def render(self, steps, total_rewards):
        moving_avr_rewards = np.divide(moving_average(total_rewards, self.config.window_reward_test), self.config.max_reward_test)
        # moving_avr_rewards = np.multiply(moving_average(total_rewards, self.config.window_reward_test), 1000)

        self.line_test.set_data(steps, moving_avr_rewards)
        self.ax_test.draw_artist(self.ax_test.patch)
        self.ax_test.draw_artist(self.line_test)
        self.fig_test.canvas.update()
        self.fig_test.canvas.flush_events()


# Display showing the results of one test
class OneTestDisplays:
    def __init__(self, config):
        self.config = config

    def plot_one_test(self, steps_rewards, total_rewardss, total_rewards_mean, steps_losss, total_losss, losss_mean, steps_qs, total_qs, qs_mean,
                      steps_rewards_test, total_rewardss_test, total_rewards_mean_test, title):
        fig, (ax4, ax1, ax2, ax3) = pp.subplots(4, figsize=(10, 40))

        for i in range(self.config.iterations):
            ax1.plot(steps_rewards[i], total_rewardss[i], 'c--')
        ax1.plot(steps_rewards[0], total_rewards_mean, label='Mean reward')
        # ax1.plot([0, self.config.max_step], [self.config.max_reward / 2, self.config.max_reward / 2],
        #          label='Half max reward')
        ax1.legend()
        ax1.set_title('Training Reward')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Moving average ({})'.format(self.config.window_reward_training))

        for i in range(self.config.iterations):
            ax2.plot(steps_losss[i], total_losss[i], 'c--')
        ax2.plot(steps_losss[0], losss_mean)
        ax2.set_title('Loss')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Moving average ({}) loss'.format(self.config.window_loss))

        for i in range(self.config.iterations):
            ax3.plot(steps_qs[i], total_qs[i], 'c--')
        ax3.plot(steps_qs[0], qs_mean)
        ax3.set_title('Q')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Moving average ({}) Q'.format(self.config.window_q))

        for i in range(self.config.iterations):
            ax4.plot(steps_rewards_test[i], total_rewardss_test[i], 'c--')
        ax4.plot(steps_rewards_test[1], total_rewards_mean_test)
        ax4.set_title('Test Reward')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Moving average ({}) reward'.format(self.config.window_reward_test))

        pp.savefig('figs/{}.png'.format(title))


# Display showing moving averages
class MeanDisplays:
    def __init__(self, config):
        self.config = config

    def plot_means(self, sr, mr, sl, ml, sq, mq, sc, mc, labels):
        fig, ((ax4, ax1),(ax2, ax3)) = pp.subplots(2,2, figsize=(20, 20))
        colors = list(pp.cm.tab10(np.arange(10))) + ["crimson", "indigo", "lavender", "darkcyan", "cornflowerblue", "tomato"]

        for i in range(len(mr)):
            ax1.plot(sr[i], mr[i], label=labels[i], color=colors[i])
        # ax1.plot([0, config.max_step], [config.max_reward, config.max_reward],
        #          label='Max reward')
        ax1.legend()
        ax1.set_title('Training Reward')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Moving average ({}) reward'.format(self.config.window_reward_training))

        for i in range(len(ml)):
            ax2.plot(sl[i], ml[i], label=labels[i], color=colors[i])
        ax2.legend()
        ax2.set_title('Loss')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Moving average ({}) loss'.format(self.config.window_loss))

        for i in range(len(mq)):
            ax3.plot(sq[i], mq[i], label=labels[i], color=colors[i])
        ax3.legend()
        ax3.set_title('Q')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Moving average ({}) Q'.format(self.config.window_q))

        for i in range(len(mc)):
            ax4.plot(sc[i], mc[i], label=labels[i], color=colors[i])
        ax4.legend()
        ax4.set_title('Test Reward')
        ax4.set_xlabel('Fraction of test')
        ax4.set_ylabel('Moving average ({}) reward'.format(self.config.window_reward_test))

        pp.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)

        pp.savefig('figs/{}_means.png'.format(self.config.title))


def moving_average(list, n):
    mov_avr = []
    cumsum = np.cumsum(list)
    for j in range(len(cumsum)):
        low = max(j-n,0)
        length = max(min(j,n),1)
        mov_avr.append((cumsum[j]-cumsum[low])/length)
    return mov_avr