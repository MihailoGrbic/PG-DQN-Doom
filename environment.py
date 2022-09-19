import numpy as np
import time

from vizdoom import DoomGame

def create_environment(config_path="scenarios/basic.cfg", scenario_path="scenarios/basic.wad", seed=None):
    game = DoomGame()
    if seed is not None: game.set_seed(seed)

    game.load_config(config_path)
    game.set_doom_scenario_path(scenario_path)
    game.init()
    
    # The environment expects one hot encoded actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    
    return game, possible_actions
       
def test_environment(config_path = "scenarios/basic.cfg", scenario_path = "scenarios/basic.wad"):
    game, actions = create_environment(config_path, scenario_path)

    episodes = 10
    for _ in range(episodes):
        game.new_episode()

        while not game.is_episode_finished():
            state = game.get_state()
            # img = state.screen_buffer 
            # misc = state.game_variables

            action = actions[np.random.choice(len(actions))]
            reward = game.make_action(action)

            print("Action:", action)
            print ("Action reward:", reward)
            time.sleep(0.02)

        print ("Episode result:", game.get_total_reward())
        time.sleep(2)
    game.close()

if __name__ == "__main__":
    test_environment("scenarios/health_gathering.cfg", "scenarios/health_gathering.wad")
    