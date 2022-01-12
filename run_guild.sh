#guild run qlearning:train epochs=[50,500] alpha=[0.001,1] gamma=[0.001,1] epsilon=[0.001,0.01,0.1,1]
#guild run qlearning:train epochs_training=[3000] steps_max_training=[1000] steps_max_testing=[1000] \
#  alpha=linspace[0.35:.99:4] gamma=linspace[0.001:0.99:4] epsilon=linspace[0.3:0.99:4]

#~/.conda/envs/rtx/bin/guild run dq:train training.batch_size=[128,512] training.learning_rate=[0.001,0.004] rl.max_steps_per_episode=100 rl.num_episodes=[10]

#~/.conda/envs/rtx/bin/guild run dq:train \
#  training.batch_size=[512,1024] training.learning_rate=[0.04] training.warmup_episode=[100,1000,2500] \
#  rl.max_steps_per_episode=200 rl.num_episodes=[2500]

#~/.conda/envs/rtx/bin/guild run dq:train \
#  training.batch_size=[512] training.learning_rate=[0.04] training.warmup_episode=[100,2500] \
#  rl.max_steps_per_episode=200 rl.num_episodes=[4500]

#~/.conda/envs/rtx/bin/guild run dq:train \
#  training.batch_size=[1024] training.learning_rate=[0.04] training.warmup_episode=[100,2500] \
#  rl.max_steps_per_episode=200 rl.num_episodes=[4500]


#~/.conda/envs/rtx/bin/guild run dq:train \
#  training.batch_size=[512] training.learning_rate=[0.001] training.warmup_episode=[100,2500] \
#  rl.max_steps_per_episode=200 rl.num_episodes=[2500]


~/.conda/envs/rtx/bin/guild run dq:train \
  training.batch_size=[1024,2048] training.learning_rate=[0.004] training.warmup_episode=[100] \
  rl.max_steps_per_episode=200 rl.num_episodes=[2500]
