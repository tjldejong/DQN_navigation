from agent import Agent
from learner import Learner
from replay_memory import ReplayMemory

import environment
import displays
import time

from multiprocessing import Process, Queue, Manager


def thread_agent(config, road_environment, memory_queue, environment_queue, agent_performance_queue, weights, lock_weights):
    agent = Agent(config, road_environment, environment_queue, agent_performance_queue, memory_queue, 'agent_1')
    agent.train(weights, lock_weights)
    return


def thread_learner(config, batch_queue, learner_performance_queue, update_p_queue, weights, lock_weights):
    learner = Learner(config, batch_queue, learner_performance_queue, update_p_queue)
    learner.learn(weights, lock_weights)
    return


def thread_memory(config, memory_queue, batch_queue, update_p_queue, priority_environment_queue):
    memory = ReplayMemory(config, memory_queue, batch_queue, update_p_queue, priority_environment_queue)
    memory.loop()
    return


def thread_agent_test(config, test_environment, test_environment_queue, test_agent_performance_queue, weights, lock_weights, memory_queue):
    agent = Agent(config, test_environment, test_environment_queue, test_agent_performance_queue, memory_queue, 'agent_test', test_environment_queue)
    agent.test(weights, lock_weights)
    return


def thread_agent_priority(config, test_environment, priority_environment_queue, memory_queue, test_environment_queue, weights, lock_weights):
    agent = Agent(config, test_environment, priority_environment_queue, None, memory_queue, 'agent_priority', test_environment_queue)
    agent.create_prioritized_memories(weights, lock_weights)
    return


def thread_game_display(road_environment, environment_queue):
    display_game = displays.GameDisplay(road_environment)
    while True:
        # print(environment_queue.qsize())  # Used to check if display can keep up
        road_environment = environment_queue.get()
        display_game.render(road_environment)


def thread_test_display(test_environment, test_environment_queue):
    display_test = displays.GameDisplay(test_environment)
    while True:
        test_environment = test_environment_queue.get()
        display_test.render(test_environment)


def one_run(config):
    test_environment = environment.EnvRoad(config)
    road_environment = environment.EnvRoad(config)

    memory_queue = Queue()
    update_p_queue = Queue()
    batch_queue = Queue()

    environment_queue = Queue()
    test_environment_queue = Queue()

    agent_performance_queue = Queue()
    learner_performance_queue = Queue()
    test_agent_performance_queue = Queue()

    priority_environment_queue = Queue()

    manger_weights = Manager()
    weights = manger_weights.dict()
    lock_weights = manger_weights.Lock()

    p_agent = Process(target=thread_agent, args=(config, road_environment, memory_queue, environment_queue, agent_performance_queue, weights, lock_weights))
    p_learn = Process(target=thread_learner, args=(config, batch_queue, learner_performance_queue, update_p_queue, weights, lock_weights))
    p_mem = Process(target=thread_memory, args=(config, memory_queue, batch_queue, update_p_queue, priority_environment_queue))
    p_agent_test = Process(target=thread_agent_test, args=(config, test_environment, test_environment_queue, test_agent_performance_queue, weights, lock_weights, memory_queue))
    p_agent_priority = Process(target=thread_agent_priority, args=(config, test_environment, priority_environment_queue, memory_queue, test_environment_queue, weights, lock_weights))


    start_time = time.time()
    p_agent.start()
    p_mem.start()
    p_learn.start()
    p_agent_test.start()
    p_agent_priority.start()

    if config.display:
        p_display_game = Process(target=thread_game_display, args=(road_environment, environment_queue))
        p_display_test = Process(target=thread_test_display, args=(test_environment, test_environment_queue))
        p_display_game.start()
        p_display_test.start()

        display_agent = displays.AgentDisplays(config)
        display_learner = displays.LearnerDisplays(config)
        display_test = displays.TestDisplays(config)

    learner_performance = [[],[],[]]
    agent_performance = [[],[]]
    test_agent_performance = [[],[]]

    end = False

    # Display all performance metrics
    while True:
        while agent_performance_queue.qsize() > 0:
            agent_performance = agent_performance_queue.get()
            if len(agent_performance[0]) == len(agent_performance[1]) and config.display:
                display_agent.render(agent_performance[0], agent_performance[1])
            if len(agent_performance[0]) > config.max_step:
                end = True

        while learner_performance_queue.qsize() > 0:
            learner_performance = learner_performance_queue.get()
            if len(learner_performance[0]) == len(learner_performance[1]) == len(learner_performance[2]) and config.display:
                display_learner.render(learner_performance[0], learner_performance[1], learner_performance[2])

        while test_agent_performance_queue.qsize() > 0:
            test_agent_performance = test_agent_performance_queue.get()
            if len(test_agent_performance[0]) == len(test_agent_performance[1]) and config.display:
                display_test.render(test_agent_performance[0], test_agent_performance[1])

        if end:
            if config.display:
                displays.close_all()
                p_display_game.terminate()
                p_display_test.terminate()
            break

    p_agent.terminate()
    p_mem.terminate()
    p_learn.terminate()
    p_agent_test.terminate()
    p_agent_priority.terminate()
    print('Average time per step: ', str((time.time()-start_time)/config.max_step))

    return agent_performance[0], agent_performance[1], learner_performance[0], learner_performance[1], \
           learner_performance[0], learner_performance[2], test_agent_performance[0], test_agent_performance[1]

