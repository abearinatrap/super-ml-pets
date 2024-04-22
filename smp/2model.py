from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sapai_gym import SuperAutoPetsEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sapai_gym.opponent_gen.opponent_generators import random_opp_generator, biggest_numbers_horizontal_opp_generator
from sapai_gym.ai import baselines
from sapai import *
from sapai.shop import *
import logging as log
from .utils import opponent_generator, get_screen_scale, kill_process


def run(ret):
    log.info("INITIALIZATION [self.run]: Loading Model")
    model1 = MaskablePPO.load(ret.infer_model, custom_objects=custom_objects)
    model2 = MaskablePPO.load(ret.infer_mode_2, custom_objects=custom_objects)

    log.info("INITIALIZATION [self.run]: Create SuperAutoPetsEnv Object")
    env1 = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)
    env2 = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)
    obs1 = env1.reset()
    obs2 = env2.reset()

    # env auto generates pets and food in shop slot for both

    action_masks = get_action_masks(env1)
    obs = env1._encode_state()

    # run until battle

    action1, _states = model1.predict(obs, action_masks=action_masks, deterministic=True)
    s1 = env1._avail_actions()
    # converting to an integer to avoid causing unhashable a TypeError
    action1 = int(action1) 

    return