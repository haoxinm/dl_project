from multiprocessing.spawn import freeze_support

from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class

def main():
    config = get_config('ppo_replay_pointnav.yaml', None)

    env = construct_envs(config, get_env_class(config.ENV_NAME))
    # env = baseline_registry.get_env(config.ENV_NAME)
    env.render()


if __name__ == "__main__":
    freeze_support()
    main()
