# tum-adlr-ws22-06
Advanced Deep Learning for Robotics project: sim2real problem in the context of flying vehicles

# How the configuration works
`conf\config.yaml` contains all the shared paramenters. \
`conf\agent\` contains agent-specific parameters. 

You can select the required agent parameters by specifying `defaults` section (important - only AFTER `_self_` to overwrite default parameters) 

If you don't want to edit configs to run the script with the specific action - you can specify it from the run cmd:
```
python baseline.py agent=ddpg
```

Any key, written to an agent's configuraion, will ovewrite defaults or will be appended to them. 

If you want create an agent-specific training configuration, you have to create file inside `conf\training\` folder with the same name as the specific `agent` file, like `conf\training\sac2.yaml`

# How to configure randomization of the env
If you want to start the training without any randomization, set `env.random_type='Fixed'` in `conf\config.yaml`. \
To configure parameters of the fixed env, edit the parameters with prefix `default_`. \
To turn on randomization, set `env.random_type='Uniform'`, and tune lower and upper bounds of the parameters with corresponding configs. \
To pass the env state (gravity, enable_wind, wind_power, turbulence_power) to an agent, set `training.pass_env_parameters=True`. \
They will be normalized, according to lower and upper bounds, and added to state (see `StateInjectorWrapper` from `EnvironmentRandomizer.py`)
 