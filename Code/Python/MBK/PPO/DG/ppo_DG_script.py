#%%
import importlib.util
import sys
#%%
#%%
# package_name = 'Env_DG'
# if package_name in sys.modules:
#     print('Package is already imported')
# else:
#     print('importing package')
#     from Env_DG import MassSpringDamperEnv
#     print('Package is imported')
from Env_DG import MassSpringDamperEnv

#%%
# package_name = 'PPO_utilz_DG'
# if package_name in sys.modules:
#     print('Package is already imported')
# else:
#     print('importing package')
#     from PPO_utilz_DG import *
#     print('Package is imported')
# package_name = 'PPO_DG'
# if package_name in sys.modules:
#     print('Package is already imported')
# else:
#     print('importing package')
#     from PPO_DG import PPO
#     print('Package is imported')
from PPO_utilz_DG import *
from PPO_DG import PPO
#%%
# Define constants
HID = 64
L = 2
GAMMA = 0.99
SEED = 0
STEPS = 4000
EPOCHS = 216
EXP_NAME = 'ppo'

# Use the constants directly



logger_kwargs = setup_logger_kwargs(EXP_NAME, SEED)

ppo = PPO(MassSpringDamperEnv(), ac_kwargs=dict(hidden_sizes=[HID] * L), gamma=GAMMA,
            seed=SEED, steps_per_epoch=STEPS, epochs=EPOCHS,
            logger_kwargs=logger_kwargs)

# test PPO agent (not trained)
ppo.test(deterministic=False)

#%%
ppo.train()
#%%
