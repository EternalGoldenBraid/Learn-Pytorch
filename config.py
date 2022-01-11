class Config:
    class training:
        batch_size = 128
        learning_rate = 0.001
        loss = "huber"
        num_episodes = 10000
        train_steps = 1000000
        warmup_episode = 100
        save_freq = 1000

    class optimizer:
        name = "adam"
        lr_min = 0.0001
        lr_decay = 5000

    class rl:
        gamma = 0.99
        max_steps_per_episode = 200
        target_model_update_freq = 20
        memory_capacity = 50000
        num_episodes = 2000
    
    class epsilon:
        max_epsilon = 1
        min_epsilon = 0.1
        decay_epsilon = 600
