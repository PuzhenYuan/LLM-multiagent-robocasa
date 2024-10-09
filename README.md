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
5. install latest ```openai``` module to support LLM settings
```
pip install openai
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

## Multitask Policy Validation

1. Validate a single-task trained agent in a multi-task one-agent environment.
Can specify different environments and command languages.
Directly reset the joint positions of the robot arm in simulation data whenever a task is completed.
```
python robomimic/robomimic/scripts/run_trained_multitask_agent_reset.py \
--agent path/to/ckpt.pth \
--env multitask_oneagent_env \
--env_lang "task0_lang, task1_lang" # follow this format \
--renderer "mujoco" # "mujoco" or "mjviewer"
```

2. Validate a single-task trained agent in a multi-task two-agent environment.
Can specify different environments and command languages.
Directly reset the joint positions of the robot arm in simulation data whenever a task is completed.
```
python robomimic/robomimic/scripts/run_trained_multitask_twoagent_reset.py \
--agent path/to/ckpt0.pth path/to/ckpt1.pth \
--env multitask_twoagent_env # any registered env starts with "TwoAgent" \
--env_lang "agent0 task0_lang, agent1 task1_lang" # follow this format \
--renderer "mjviewer" # "mujoco" or "mjviewer"
```

## Controller

Default PID planners for single-tasks:
* Navigation
* Pick
* Place

Can be called automatically in teleoperate script, facilitating demonstration collection for tasks that lack released mimicgen demonstrations.

These planners are stored in `robocasa/robocasa/utils/planner`, which are also used in `robocasa/robocasa/utils/controller_dict.py` for language-guided robot control.

## Language Instruction Control

1. Using standardized language instructions to guide robot manipulation or navigation.
The input language command should match keys in `robocasa/robocasa/utils/controller_dict.py`, and will call either hardcode planner or checkpoint policy to control the robot.
```
python robomimic/robomimic/scripts/run_controlled_multitask_agent.py \
--agent path/to/ckpt.pth # also need to provide a path \
--env multitask_agent_env \
--renderer "mjviewer" # "mujoco" or "mjviewer"
```

2. Using standardized language instructions to guide two-robot manipulation or navigation.
The input language command should match keys in `robocasa/robocasa/utils/controller_dict.py`, and will call either hardcode planner or checkpoint policy to control the robot.
```
python robomimic/robomimic/scripts/run_controlled_multitask_twoagent.py \
--agent path/to/ckpt0.pth path/to/ckpt1.pth # also need to provide paths \
--env multitask_twoagent_env # any registered env starts with "TwoAgent" \
--renderer "mjviewer" # "mujoco" or "mjviewer"
```

## LLM Guided Control

1. Call an openai client to generate plans and commands to guide robot manipulation or navigation.
Based on the language instruction control script and settings above.
```
python robomimic/robomimic/scripts/run_llm_multitask_agent.py \
--agent path/to/ckpt.pth # also need to provide a path \
--env multitask_agent_env \
--renderer "mjviewer" # "mujoco" or "mjviewer"
```
