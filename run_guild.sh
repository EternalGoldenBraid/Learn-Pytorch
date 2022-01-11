#guild run qlearning:train epochs=[50,500] alpha=[0.001,1] gamma=[0.001,1] epsilon=[0.001,0.01,0.1,1]
#guild run qlearning:train epochs_training=[3000] steps_max_training=[1000] steps_max_testing=[1000] \
#  alpha=linspace[0.35:.99:4] gamma=linspace[0.001:0.99:4] epsilon=linspace[0.3:0.99:4]

guild run dq:train batch_size=[128,512] learning_rate=[0.001,0.004] max_steps_per_episode=100 \
  		num_episodes=[10,100]

#guild run qlearning:train epochs_training=[100,1000] steps_max_training=[500] epsilon=0.4
