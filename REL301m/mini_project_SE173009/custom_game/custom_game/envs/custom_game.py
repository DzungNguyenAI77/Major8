import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
import pygame
from pygame import gfxdraw

class CustomGame(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600 * 3
        self.screen_height = 400 * 3
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action: int):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Range of the gap
        gap_start = self.max_position * 0.7 
        gap_end = self.max_position * 0.75  


        # Check sufficient velocity
        if gap_start <= position <= gap_end and abs(velocity) < self.goal_velocity:
            print("Tank did not have enough velocity to cross the gap!")
            terminated = True
            reward = -100  
        else:
            terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
            reward = -1.0
            print("Tank passed the gap and achieve the goal!")


        # Update state
        self.state = (position, velocity)
        
        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        initial_velocity = 0
        self.state = np.array([self.np_random.uniform(low=low, high=high), initial_velocity])

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise ImportError(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # pos = self.state[0]
        # vel = self.state[1]

        xs = np.linspace(self.min_position, self.max_position, 100)

       
        # Gap
        break_start = int(self.screen_width * 0.7)
        break_end = int(self.screen_width * 0.75)

        
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        
        left_segment = [point for point in xys if point[0] < break_start]
        right_segment = [point for point in xys if point[0] > break_end]

        
        pygame.draw.aalines(self.surf, (225, 0, 0), False, left_segment)
        pygame.draw.aalines(self.surf, (225, 0, 0), False, right_segment)

        clearance = 10

        # Tank
        top_length = 60
        base_length = 80
        body_height = 20
        body_x = int((self.state[0] - self.min_position) * scale)
        body_y = int(clearance + self._height(self.state[0]) * scale) + 15 + body_height // 2

        body_coords = [
            (body_x - top_length // 2, body_y),
            (body_x + top_length // 2, body_y),
            (body_x + base_length // 2, body_y - body_height),
            (body_x - base_length // 2, body_y - body_height),
        ]
        gfxdraw.aapolygon(self.surf, body_coords, (0, 100, 0))
        gfxdraw.filled_polygon(self.surf, body_coords, (34, 139, 34))

        turret_radius = 10
        turret_x = body_x
        turret_y = body_y + turret_radius
        gfxdraw.aacircle(self.surf, turret_x, turret_y, turret_radius, (0, 0, 0))
        gfxdraw.filled_circle(self.surf, turret_x, turret_y, turret_radius, (34, 100, 34))

        cannon_length = 30
        cannon_width = 4
        cannon_start = (turret_x, turret_y)
        cannon_end = (turret_x + cannon_length, turret_y)
        pygame.draw.line(self.surf, (0, 0, 0), cannon_start, cannon_end, cannon_width)

        wheel_radius = 5
        num_wheels = 7
        wheel_spacing = base_length // (num_wheels + 1)
        wheel_y = int(clearance + self._height(self.state[0]) * scale)
        for i in range(num_wheels):
            wheel_x = body_x - base_length // 2 + (i + 1) * wheel_spacing
            gfxdraw.aacircle(self.surf, wheel_x, wheel_y, wheel_radius, (128, 128, 128))
            gfxdraw.filled_circle(self.surf, wheel_x, wheel_y, wheel_radius, (128, 128, 128))

        # Flag
        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
