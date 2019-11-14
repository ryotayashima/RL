from gym.envs.registration import register

register(
    id='LiftingODE-v0',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 5000},
    reward_threshold=4750.0,
    entry_point='myenv.env:LiftingODEEnv'
)
