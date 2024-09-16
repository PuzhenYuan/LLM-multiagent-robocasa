# LLM-multiagent-robocasa
Implement multiagent settings in robocasa and empower them with collaborative skills through LLM.

<img src="photo/corner.jpg" width="50%"><img src="photo/wall.jpg" width="50%">

## Installation
1. set up conda environment, python 3.9 is recommended
```
conda create -c conda-forge -n robocasa python=3.9
conda activate robocasa
```   
2. install ```robosuite``` module in editable mode, branch：robocasa_v0.1
```
cd robosuite
pip install -e .
```
3. install ```robocasa``` module in editable mode, branch：main
```
cd robocasa
pip install -e .
conda install -c numba numba -y
python robocasa/scripts/download_kitchen_assets.py
python robocasa/scripts/setup_macros.py
```
4. install ```robomimic``` module in editable mode, branch: robocasa
```
cd robomimic
pip install -e .
```

## Teleoperate
Teleoperate manually in this script. 
Use a keyboard to control and a mouse to adjust the rendering camera if the renderer is mjviewer (default).
Task includes single-task, multi-task, one-agent-task, two-agent-task. 
This script can also collect trajectories for imitation learning.
```
cd robocasa
python robocasa/demos/multi_teleop_test.py 
```

## Controller
Default PID controllers for simgle-tasks:
* NavigateKitchen
* PnPCounterToCounter

Can be called automatically in teleoperate script, facilitating demonstration collection for tasks which lack released mimicgen demonstrations.

## Validation
`cd robomimic`

1. Validate a single-task trained agent in the training environment.
Use the standard way provided by robomimic project. 
```
python robomimic/scripts/run_trained_singletask_agent_template.py \
--agent path/to/ckpt.pth
```

2. Validate a single-task trained agent in a single-task one-agent environment.
Can specify different environments and command languages.
```
python robomimic/scripts/run_trained_singletask_agent.py \
--agent path/to/ckpt.pth \
--env singletask_oneagent_env \
--env_lang "task_lang"
```

3. Validate a single-task trained agent in a multi-task one-agent environment.
Can specify different environments and command languages.
```
python robomimic/scripts/run_trained_multitask_agent.py \
--agent path/to/ckpt.pth \
--env multitask_oneagent_env \
--env_lang "task0_lang, task1_lang" # follow this format
```

4. Validate a single-task trained agent in a multi-task one-agent environment.
Can specify different environments and command languages.
Directly reset the joint positions of the robot arm in simulation data whenever a task is completed.
```
python robomimic/scripts/run_trained_multitask_agent_reset.py \
--agent path/to/ckpt.pth \
--env multitask_oneagent_env \
--env_lang "task0_lang, task1_lang" # follow this format
```

5. Validate a single-task trained agent in a multi-task two-agent environment.
Can specify different environments and command languages.
```
python robomimic/scripts/run_trained_multitask_twoagent.py \
--agent path/to/ckpt.pth \
--env multitask_twoagent_env \
--env_lang "agent0 task0_lang, agent1 task1_lang" # follow this format
```

6. Validate a single-task trained agent in a multi-task two-agent environment.
Can specify different environments and command languages.
Directly reset the joint positions of the robot arm in simulation data whenever a task is completed.
```
python robomimic/scripts/run_trained_multitask_twoagent_reset.py \
--agent path/to/ckpt.pth \
--env multitask_twoagent_env \
--env_lang "agent0 task0_lang, agent1 task1_lang" # follow this format
```
