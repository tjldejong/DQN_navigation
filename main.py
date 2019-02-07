from test import Tests
from config import Config

if __name__ == '__main__':
    configs = [Config(), Config(), Config()]

    for c in configs:
        c.title = 'Sampling_methods_mem'
        c.iterations = 2

    configs[0].sampling_method = 'prioritized_mem'
    configs[0].imporatance_sampling = True
    configs[0].last = True
    configs[0].label = 'prioritized_mem_is_last'

    configs[1].sampling_method = 'prioritized_mem'
    configs[1].imporatance_sampling = True
    configs[1].last = False
    configs[1].label = 'prioritized_mem_is'

    configs[2].sampling_method = 'prioritized_mem'
    configs[2].imporatance_sampling = False
    configs[2].last = False
    configs[2].label = 'prioritized_mem'

    Tests().test(configs)

