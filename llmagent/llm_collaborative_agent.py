import os
import openai
import re
from pprint import pprint
from termcolor import colored


class LMCoAgent:
    """
    language model agent to generate plan and command for robot in robocasa environment
    """
    def __init__(self, goal, env_info, api_key, base_url, model="gpt-3.5-turbo", default_headers=None, id=0):
        """
        initialize LMAgent with api_key, base_url and default_headers
        """
        
        # unset all_proxy and ALL_PROXY to avoid proxy issues
        os.environ['all_proxy'] = ""
        os.environ['ALL_PROXY'] = ""
        
        # passing parameters
        self.id = id
        self.goal = goal
        self.env_info = env_info
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.default_headers = default_headers
        self.client = openai.OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url, 
            default_headers=self.default_headers
        )
        
        # initialize system prompt and first user prompt, using chain of thought
        # TODO: tailor the prompt to suit multiagent collaborative environment
        self.history_messages = [
            {
                "role": "system",
                "content": "You are a kitchen robot high-level planner, and you are able to plan long-horizon and multi-stage tasks step by step given current step info. " \
                    + "Every time you output plan and execution given history messages, first examine the history and deteremine whether to change previous task plans. " \
                    + "And then you output the current optimal plan using chain of thought and the command for the next step of execution. " \
                    + "Follow this format when output: Plan: [step0, step1, step2, ...], Execution: step0"
            },
            {
                "role": "user",
                "content": \
"""
Example: 

Environment infomation:
Available fixtures in environment: sink, stove, fridge, microwave
Available objects in environment: vegetable, container
Available commands: wait, pick up (object), place to (object or fixture), navigate to (object or fixture), open microwave door
Goal: pick up vegetable from counter and place it to container in the microwave

Your output at task 0: 
Plan: [navigate to microwave, open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave, place to container], Execution: navigate to microwave

Success: task 0 navigate to microwave succeeded

Your output at task 1: 
Plan: [open microwave door, navigate to vegetable, pick up vegetable, navigate to microwave, place to container], Execution: open microwave door
...

""" \
                + "In this task, environment infomation: \n" + self.env_info + "Goal: " + self.goal + "\n"
            }
        ]

    def update_messages(self, message):
        """
        message (str) should be environment feedback message containing success flag or failure reason,
        this method aims to append this feedback message to self.history_messages
        """
        assert message.startswith(("Success", "Error", "Failure", "Warning"))
        self.history_messages.append(
            {
                "role": "user",
                "content": message
            }
        )
    
    def get_command(self, verbose=False):
        """
        call the self.client to output new plan and command based on self.history_messages
        should output future plan and next step command
        """
        
        # refine message prompt
        messages = self.history_messages
        assert messages[-1]["role"] == "user"
        messages[-1]["content"] += \
        "Please output plan for completing tasks and commands for the next step of execution, follow this format: " \
        + "Plan: [step0, step1, step2, ...], Execution: step0"
        
        # get client response
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        content = completion.choices[0].message.content
        if verbose:
            print(content)
        
        # update history messages
        self.history_messages.append(
            {
                "role": "assistant",
                "content": content
            }
        )
        
        # extract plan and execution using re
        plan_match = re.search(r"Plan: \[(.*?)\]", content)
        plan = plan_match.group(1).split(", ") if plan_match else []
        execution = plan[0]
        
        return plan, execution
    
    def show_history(self):
        """
        show agent message history
        """
        for message in self.history_messages:
            print(colored(message["role"] + ": ", "cyan"))
            print(message["content"])
            print()


if __name__ == "__main__":
    goal = "pick up obj from counter and place it to container in the microwave"
    env_info = 'Available fixtures in environment: fridge, sink, opencabinet, '\
            + 'dishwasher, stove, microwave, coffeemachine, toaster\n' \
            + 'Available objects in environment: obj, obj_container, container, ' \
            + 'distr_counter\n' \
            + 'Available commands: wait, reset arm, pick up, place to, navigate ' \
            + 'to, navigation, open microwave door, pick the object from the ' \
            + 'counter and place it in the microwave\n'
    lmagent = LMCoAgent(goal=goal, env_info=env_info, api_key="", base_url="")
    lmagent.show_history()
    plan, command = lmagent.get_command()
    print(plan)
    print(command)