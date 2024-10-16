This contains all the wrappers that gives additional functionalities to the base agent.

We are looking for:
- PlanningWrappers: How to plan and execute the plan, self-correct plan if there are any errors
- ReflectionWrappers: How to reflect upon what has been learned during task / across task and consolidate info for the next task
- VerifierWrappers: How to verify agent's outputs and correct them accordingly
- ConversationWrappers: How to converse with the agent
- MultiAgentWrappers: How multiple agents can converse with one another

Any other Wrapper is fine. This helps to expand upon the characteristics of the Agent without changing the base code.

See `ExampleWrapper.py` for an example of what the Notebook should contain

If it is good and sufficiently used by others, it will be ported over to the main TaskGen repo with acknowledgement of you as the author :)
