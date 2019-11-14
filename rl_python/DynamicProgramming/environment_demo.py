import random
from environment import Environment

class Agent():

    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        return random.choice(self.actions)

def main():

    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    for i in range(2):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # print(env.agent_state)
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print("Episode {}: Agent gets {} reward.".format(i, total_reward))

if __name__ == "__main__":
    main()
            
