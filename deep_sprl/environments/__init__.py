from gymnasium.envs.registration import register

# Colored Sokoban ('Colorbox-v1') is registered by the gym_sokoban package.
register(
    id='ButtonMaze-v1',
    max_episode_steps=100,
    entry_point='deep_sprl.environments.button_maze:ButtonMazeEnv'
)
