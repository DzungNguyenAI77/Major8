from gymnasium.envs.registration import register

register(
    id='CustomGame-v0',
    entry_point='custom_game.envs:CustomGame'
)