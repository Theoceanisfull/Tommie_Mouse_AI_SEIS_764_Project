# Activate Python Enviroment 
In Terminal: source rl_env/bin/activate
# Read maze_env.py file and test the enviroment
Read the maze_env python file to understand the enviroment. You can adjust the enviroment as needed.
Please see OpenAI Gym for documentation at https://gymnasium.farama.org 

Run the test_maze_env pyfile to see how the enviroment works.
In Terminal: python test_maze_env.py

This will demonstrate the maze_env without any-real learning as the action are random
# Q Learning on a single fixed maze 
This will update the q table and give reward for a set of action in a state.
For example, position 0 is the first state, the agent can take the action up, down, right and left. 
This goes on for all the position the agent can move until it reaches it's ultimate goal the target cheese at the end of the maze. 

