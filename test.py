import subprocess
import sys


sys.path.extend(['/root/dl_project', '/root/dl_project'])
# subprocess.call(['conda', 'activate', 'habitat'])
subprocess.call(['/root/dl_project/run.py', '--exp-config', 'ddppo_tamer_pointnav.yaml',
                 '--run-type', 'train'])
# subprocess.call(['/root/dl_project/keyboard_agent.py'])
