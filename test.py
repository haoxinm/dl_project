import subprocess
import sys


sys.path.extend(['/root/dl_project', '/root/dl_project'])
# subprocess.call(['conda', 'activate', 'habitat'])
subprocess.call(['/root/dl_project/run.py', '--exp-config', 'ppo_replay_pointnav.yaml',
                 '--run-type', 'train'])
