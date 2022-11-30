# tum-adlr-ws22-06
Advanced Deep Learning for Robotics project: sim2real problem in the context of flying vehicles

# How the configuration works
`conf\config.yaml` contains all the shared paramenters. \
`conf\agent\` contains agent-specific parameters. 

You can select the reguired agent parameters by specifying `defaults` section (important - only AFTER `_self_` to overwrite default parameters) 

If you don't want to edit configs to run the script with the specific action - you can specify it from the run cmd:
```
python baseline.py agent=ddpg
```

Any key, written to an agent's configuraion, will ovewrite defaults or be appended to them. 

If you want create an agent-specific training, you have to create file inside `training\` folder with the same name as the specific `agent` file, like `training\sac2.yaml`
