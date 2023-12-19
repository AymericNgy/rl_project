import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import torch


# [!] look how they deal with dt in RL TDs

class EnergyConverter(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        z_init = 0
        w_init = 0
        x_ss_init = 0 # [!] not good shape

        # state [z, w, x_ss].T
        self.x = np.array(z_init, w_init, x_ss_init)

        # model parameters
        # [!] not good shapes, need to know shape of x_ss
        self.A = np.zeros((3, 3), dtype=float)  # [!] to really implement
        self.B = np.zeros((3, 1), dtype=float)  # [!] to really implement
        self.c = np.zeros((3, 1), dtype=float)  # [!] to really implement

        # observation space [z, dz/dt, zeta, dzeta/dt].T
        self.observation_space = spaces.Box(low=-np.ones(4) * np.inf, high=np.ones(4) * np.inf, shape=(4, 1),
                                            dtype=np.float32)

