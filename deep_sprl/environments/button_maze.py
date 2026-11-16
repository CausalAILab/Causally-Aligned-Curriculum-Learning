from __future__ import annotations

import copy
from typing import Any

import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class Indicator(Goal):
    """A goal tile whose color can be changed (used as the exit / traffic light)."""

    def __init__(self, color="green"):
        super().__init__()
        self.color = color


class ButtonMazeEnv(MiniGridEnv):
    """
    Button Maze: a small maze with a button and an exit tile that doubles as a traffic light.

    The agent should leave through the exit on the right border only when the (hidden) true
    light is green. Before the button is pressed the true light is green with probability
    `ratio`/100 at every step and the displayed light color is random, i.e. uninformative.
    After the button is pressed the true light is always green and it is displayed.

    ## Action Space (remapped to absolute directions)

    | Num | Action                 |
    |-----|------------------------|
    | 0   | move left              |
    | 1   | move right             |
    | 2   | move up                |
    | 3   | move down              |
    | 4   | toggle (press button)  |

    ## Rewards

    -0.3 / max_steps per step, -0.1 if the agent did not move, +1 (-1) for entering the exit
    while the true light is green (red), +0.6 for pressing the button the first time and
    -0.6 for un-pressing it.

    ## Termination

    The agent enters the exit, or timeout (`max_steps`).

    ## Context (length 5 + 1 + 1 + 1 = 8; values are floored to integers by `set_context`)

    - [0:5] one bit per wall cell of the default map (1 = wall present)
    - [5]   fix button: 1 -> the button starts pressed
    - [6]   fix light: 1 (raw value > 0.25) -> the displayed light is always green and the
            true light stays random even after the button is pressed
    - [7]   agent start cell, index in [0, 10] over the 11 free cells of the map
    """

    def __init__(
        self,
        size=6,
        agent_start_dir=0,
        max_steps: int | None = None,
        render_mode="rgb_array",
        ratio=30,
        **kwargs,
    ):
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            render_mode=render_mode,
            agent_pov=False,
            highlight=False,
            **kwargs,
        )
        # Agent action -> MiniGrid action: 0 - left, 1 - right, 2 - forward (up), 7 - down, 5 - toggle
        self.action_mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 7,
            4: 5,
        }
        self.action_space = spaces.Discrete(len(self.action_mapping))

        self.LIGHT_COLOR = ['red', 'green']
        # ratio/100 chance of a green light
        self.ratio = ratio

        self.size = size
        # Flattened inner (size-2) x (size-2) map, 1 = wall cell that the context can remove
        self._default_map = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
        self._map = copy.copy(self._default_map)
        # Map the agent location index (context[-1]) to the flat index of a free cell
        self._agent_pos_mapping = {}
        ind = 0
        for i in range(len(self._map)):
            if self._map[i]:
                continue
            self._agent_pos_mapping[ind] = i
            ind += 1
        self.context_len = sum(self._map) + 3
        self._context = None

    def set_context(self, context):
        # round down and change type to int
        assert len(context) == self.context_len
        self._context = np.floor(context).astype(np.int8)
        self._context[-2] = context[-2] > 0.25
        self._map = copy.copy(self._default_map)
        self._map[self._default_map == 1] = self._context[:self.context_len - 3]

    def get_context(self):
        return copy.deepcopy(self._context)

    context = property(get_context, set_context)

    @staticmethod
    def _gen_mission():
        return "push the button and enter the exit when the light is green"

    def _gen_grid(self, width, height):
        if self._context is None:
            # no context set: default map (all walls present), agent at cell 0, button and light not fixed
            self._context = np.zeros(self.context_len, dtype=np.int8)

        # True light (hidden confounder): green with probability ratio/100
        self.color_confounder = int(self.np_random.integers(0, 100) < self.ratio)
        self.true_light_color = self.LIGHT_COLOR[self.color_confounder]

        if self._context[-3] == 1:
            self.button_status = True

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Displayed light color
        if self.button_status and self._context[-2] == 0:
            self.true_light_color = 'green'
            self.light_color = self.true_light_color
        elif self._context[-2] == 1:
            # fixed to be always green
            self.light_color = 'green'
        else:
            self.light_color = self._rand_elem(self.LIGHT_COLOR)

        # Exit / light on the right border, button (a ball) right above it
        self.grid.set(self.size - 1, self.size - 2, Indicator(self.light_color))
        if self.button_status:
            self.grid.set(self.size - 1, self.size - 3, Ball('yellow'))
        else:
            self.grid.set(self.size - 1, self.size - 3, Ball('blue'))

        # Place the agent according to the context
        self.agent_pos = np.array(self._flat_to_loc(self._agent_pos_mapping[self._context[-1]]))
        self.agent_pos += np.array([1, 1])  # map to the grid coordinate
        self.agent_pos = self.agent_pos[::-1]  # flip the row,col coord
        self.agent_dir = self.agent_start_dir

        # Place the walls according to the map
        for i, w in enumerate(self._map):
            x, y = self._flat_to_loc(i)
            if w == 1:
                self.grid.set(y + 1, x + 1, Wall())

        self.mission = self._gen_mission()

    def _flat_to_loc(self, flat_loc):
        return [flat_loc // (self.size - 2), flat_loc % (self.size - 2)]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ):
        self.button_status = False

        # self._gen_grid() is called by the parent's reset, after (re)seeding self.np_random if a seed is given
        obs, _ = super().reset(seed=seed)
        # Only reward pressing the button if it does not start pressed
        self.first_toggle = not self.button_status

        return obs, {}

    def step(self, action):
        action = self.action_mapping[action]

        # Translate absolute-direction moves into MiniGrid's turn + forward
        if action == self.actions.left:
            self.agent_dir = 2
            action = self.actions.forward
        elif action == self.actions.right:
            self.agent_dir = 0
            action = self.actions.forward
        elif action == self.actions.forward:
            self.agent_dir = 3
            action = self.actions.forward
        elif action == 7:  # down
            self.agent_dir = 1
            action = self.actions.forward

        front_cell = self.grid.get(*self.front_pos)

        prev_pos = copy.copy(self.agent_pos)
        obs, _, terminated, truncated, info = super().step(action)

        # Step penalty
        reward = -0.3 * 1 / self.max_steps

        # Not moving penalty
        reward += -0.1 * np.all(self.agent_pos == prev_pos)

        # Entering the exit: reward depends on the true light color
        if front_cell and front_cell.type == "goal" and action == self.actions.forward:
            if self.true_light_color == self.LIGHT_COLOR[1]:
                reward += 1.0
            else:
                reward += -1.0
            terminated = True

        info['success'] = terminated and (reward > 0)

        # Update button status
        if action == self.actions.toggle and front_cell and front_cell.type == "ball":
            self.button_status = abs(self.button_status - 1)
            if self.button_status:
                self.grid.set(self.size - 1, self.size - 3, Ball('yellow'))
                if self.first_toggle:
                    reward += 0.6
                    self.first_toggle = False
            else:
                self.grid.set(self.size - 1, self.size - 3, Ball('blue'))
                reward -= 0.6

        # True light: random before pressing the button (or if the light is fixed), green afterwards
        self.color_confounder = int(self.np_random.integers(0, 100) < self.ratio)
        self.true_light_color = self.LIGHT_COLOR[self.color_confounder] if (not self.button_status or self._context[-2]) else 'green'

        if self.button_status and self._context[-2] == 0:
            self.light_color = self.true_light_color
        elif self._context[-2] == 1:
            # light color is fixed to be always green
            self.light_color = 'green'
        else:
            self.light_color = self._rand_elem(self.LIGHT_COLOR)
        if not terminated:
            self.grid.set(self.size - 1, self.size - 2, Indicator(self.light_color))

        return obs, reward, terminated, truncated, info
