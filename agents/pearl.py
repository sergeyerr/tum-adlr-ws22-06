
# this is an implementation of pearl with pytorch. They also implemented sac. It is super complicated because of all
# the inheritance
# https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/pearl.py
#
# thats the original implementation but it is not using pytorch so its super painful to read
# https://github.com/katerakelly/oyster
#
# this is a implementation that uses pytorch and only implements pearl, which should make it easier to understand
# I think it is actually the same thing as the original implementation
# https://github.com/waterhorse1/Pearl_relabel

# we will need a new replay buffer structure
# we will need functions to sample these replay buffers
# meta training, meta testing must be implemented
# the networks can mostly be used from sac

