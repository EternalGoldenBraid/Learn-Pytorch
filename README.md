# Learn-Pytorch

- Transforming a Q-learning solution done during a ML class into a DQN solution.


## Notes

- Replay memory: Could this be represented by a (di)graph pruned over time to 
allow representation of old memories? Instead of a deque discarding them.
See prioritized sweeping. 
	- https://www.nature.com/articles/nature14236 Human-level control through deep reinforcement learning


- Two networks are used to avoid feedback loops due to updating weights of the network
used to update Q(s_t,a_t) which could affect Q(s_(t+1), a) and cause unwanted
divergece and oscillation. Instead a copy of the training network is used to generate
labeled data with the help of which loss is computed and training network updated.

