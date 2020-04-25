import time
from multiprocessing.spawn import freeze_support

import gym
import matplotlib
import pygame
from gym import logger
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions

try:
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt
except ImportError as e:
	logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
	plt = None

from pygame.locals import VIDEORESIZE
import pdb
from ppo_replay import draw_top_down_map

import numpy as np


def display_arr(screen, arr, video_size, transpose, obs=None, info=None):
	arr_min, arr_max = arr.min(), arr.max()
	arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
	pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
	# pyg_img = pygame.transform.scale(pyg_img, video_size)
	# screen.blit(pyg_img, (0,0))
	img = draw_top_down_map(info[0], obs[0]["heading"][0], obs[0]['depth'].shape[0])
	map_ = pygame.surfarray.make_surface(img)
	screen.blit(map_, (0, 0))
	# print(np.average(img))

	# pdb.set_trace()
	# try:
	# if obs is not None and info is not None:
	# 	# img = np.transpose(draw_top_down_map(info[0], obs[0]["heading"][0], obs[0]['depth'].shape[0]), [2, 0, 1])
	# 	map_ = pygame.surfarray.make_surface(draw_top_down_map(info[0], obs[0]["heading"][0], obs[0]['depth'].shape[0]))
	# 	screen.blit(map_, (video_size[0],video_size[1]))
	# except:
	# 	pdb.set_trace()

def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
	"""Allows one to play the game using keyboard.
	To simply play the game use:
		play(gym.make("Pong-v4"))
	Above code works also if env is wrapped, so it's particularly useful in
	verifying that the frame-level preprocessing does not render the game
	unplayable.
	If you wish to plot real time statistics as you play, you can use
	gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
	for last 5 second of gameplay.
		def callback(obs_t, obs_tp1, action, rew, done, info):
			return [rew,]
		plotter = PlayPlot(callback, 30 * 5, ["reward"])
		env = gym.make("Pong-v4")
		play(env, callback=plotter.callback)
	Arguments
	---------
	env: gym.Env
		Environment to use for playing.
	transpose: bool
		If True the output of observation is transposed.
		Defaults to true.
	fps: int
		Maximum number of steps of the environment to execute every second.
		Defaults to 30.
	zoom: float
		Make screen edge this many times bigger
	callback: lambda or None
		Callback if a callback is provided it will be executed after
		every step. It takes the following input:
			obs_t: observation before performing action
			obs_tp1: observation after performing action
			action: action that was executed
			rew: reward that was received
			done: whether the environment is done or not
			info: debug info
	keys_to_action: dict: tuple(int) -> int or None
		Mapping from keys pressed to action performed.
		For example if pressed 'w' and space at the same time is supposed
		to trigger action number 2 then key_to_action dict would look like this:
			{
				# ...
				sorted(ord('w'), ord(' ')) -> 2
				# ...
			}
		If None, default key_to_action mapping for that env is used, if provided.
	"""
	obs = env.reset()
	# pdb.set_trace()
	rendered = env.render(mode='rgb_array')

	if keys_to_action is None:
		if hasattr(env, 'get_keys_to_action'):
			keys_to_action = env.get_keys_to_action()
		# elif hasattr(env.unwrapped, 'get_keys_to_action'):
		#     keys_to_action = env.unwrapped.get_keys_to_action()
		else:
			assert False, env.spec.id + " does not have explicit key to action mapping, " + \
						  "please specify one manually"
	relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

	video_size = [rendered.shape[1]*2, rendered.shape[0]*2]
	if zoom is not None:
		video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

	pressed_keys = []
	running = True
	env_done = True

	screen = pygame.display.set_mode(video_size)
	clock = pygame.time.Clock()

	#dummy run
	# obs = env.reset()
	# action = 1
	# prev_obs = obs
	# outputs = env.step([action])
	# obs, rew, env_done, info = [list(x) for x in zip(*outputs)]
	# rendered = env.render(mode='rgb_array')
	# env_done = True

	# screen = pygame.display.set_mode([video_size[0]*2, video_size[1]*2])
	# clock = pygame.time.Clock()

	#formal run
	while running:
		if env_done:
			print(env_done)
			env_done = False
			obs = env.reset()
		else:
			action = keys_to_action.get(tuple(sorted(pressed_keys)), (-1,))[0]
			prev_obs = obs
			action = [1,1,1,2,2,2]
			print(action)
			print(env_done)
			if len(action)>=0:
				outputs = env.step(action)
				obs, rew, env_done, info = [list(x) for x in zip(*outputs)]
				pdb.set_trace()
				try:
					print("obs length:", len(obs))
					print("info length:", len(info))
				except:
					print("notworking")
				print(env_done)
				if callback is not None:
					callback(prev_obs, obs, action, rew, env_done, info)
				if obs is not None:
					# rendered = env.render(mode='rgb_array')
					rendered = obs[0]["rgb"]
					display_arr(screen, rendered, transpose=transpose, video_size=video_size, obs=obs, info=info)

		# process pygame events
		for event in pygame.event.get():
			# test events, set key states
			if event.type == pygame.KEYDOWN:
				if event.key in relevant_keys:
					pressed_keys.append(event.key)
				elif event.key == 27:
					running = False
			elif event.type == pygame.KEYUP:
				if event.key in relevant_keys:
					pressed_keys.remove(event.key)
			elif event.type == pygame.QUIT:
				running = False
			elif event.type == VIDEORESIZE:
				video_size = event.size
				screen = pygame.display.set_mode(video_size)
				print(video_size)

		pygame.display.flip()
		clock.tick(fps)
	pygame.quit()


def main():
	config = get_config('ppo_replay_pointnav.yaml', None)

	env = construct_envs(config, get_env_class(config.ENV_NAME))
	# env = baseline_registry.get_env(config.ENV_NAME)
	keys_to_action = dict()
	keys_to_action[tuple(sorted([ord('w')]))] = HabitatSimActions.MOVE_FORWARD,
	keys_to_action[tuple(sorted([ord('a')]))] = HabitatSimActions.TURN_LEFT,
	keys_to_action[tuple(sorted([ord('d')]))] = HabitatSimActions.TURN_RIGHT,
	keys_to_action[tuple(sorted([ord('\n')]))] = HabitatSimActions.STOP,
	keys_to_action[tuple(sorted([ord('\r')]))] = HabitatSimActions.STOP,
	print(keys_to_action)


	play(env, keys_to_action=keys_to_action)
	'''
	ACTIONS = env.action_spaces[0]
	SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you
	# can test what skip is still usable.
	human_agent_action = 0
	human_wants_restart = False
	human_sets_pause = False

	def key_press(key, mod):
		global human_agent_action, human_wants_restart, human_sets_pause
		if key == 0xff0d: human_wants_restart = True
		if key == 32: human_sets_pause = not human_sets_pause
		a = int(key - ord('0'))
		if a <= 0 or a >= ACTIONS: return
		human_agent_action = a

	def key_release(key, mod):
		global human_agent_action
		a = int(key - ord('0'))
		if a <= 0 or a >= ACTIONS: return
		if human_agent_action == a:
			human_agent_action = 0

	print("==========")
	print("start render")
	print("==========")
	env.render()
	env.viewer.window.on_key_press = key_press
	env.viewer.window.on_key_release = key_release
	
	def rollout(env):
		global human_agent_action, human_wants_restart, human_sets_pause
		human_wants_restart = False
		obser = env.reset()
		skip = 0
		total_reward = 0
		total_timesteps = 0
		while 1:
			if not skip:
				#print("taking action {}".format(human_agent_action))
				a = human_agent_action
				total_timesteps += 1
				skip = SKIP_CONTROL
			else:
				skip -= 1

			obser, r, done, info = env.step(a)
			if r != 0:
				print("reward %0.3f" % r)
			total_reward += r
			window_still_open = env.render()
			if window_still_open==False: return False
			if done: break
			if human_wants_restart: break
			while human_sets_pause:
				env.render()
				time.sleep(0.1)
			time.sleep(0.1)
		print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

	print("ACTIONS={}".format(ACTIONS))
	print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
	print("No keys pressed is taking action 0")

	while 1:
		window_still_open = rollout(env)
		if window_still_open==False: break
	'''


if __name__ == "__main__":
	freeze_support()
	main()
