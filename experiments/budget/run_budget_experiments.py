#!/usr/bin/env python3

from os import makedirs
from MCTS_bot_budget_10 import *
from MCTS_bot_budget_20 import *
from MCTS_bot_budget_50 import *
from MCTS_bot_budget_100 import *
from MCTS_bot_budget_200 import *
from MCTS_bot_budget_500 import *

from random_bot import *

import botbowl as botbowl
import time as time

config = botbowl.load_config("web")
config.competition_mode = False
config.pathfinding_enabled = True
config.debug_mode = False
# We don't need all the rules
ruleset = botbowl.load_rule_set(config.ruleset, all_rules=False)
arena = botbowl.load_arena(config.arena)
home = botbowl.load_team_by_filename("human", ruleset)
away = botbowl.load_team_by_filename("human", ruleset)

mcts_configs = ['budget-10', 'budget-20',
                'budget-50', 'budget-100', 'budget-200', 'budget-500']

reps = 50

for mcts_config in mcts_configs:
    winners = []
    # Play 10 games
    for _ in range(reps):
        home_agent = botbowl.make_bot(f'MCTS-bot-{mcts_config}')
        home_agent.name = "MCTS"
        away_agent = botbowl.make_bot('e-random-bot')
        away_agent.name = "RANDOM"
        game = botbowl.Game(mcts_config+str(_), home, away, home_agent, away_agent,
                            config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        try:
            print("Starting game", mcts_config, (_+1))
            start = time.time()
            game.init()
            end = time.time()
            winner = game.get_winner()
            name = "None"
            if winner is not None:
                name = winner.name
            print("Finished in:", end - start,
                  "winner: ", name)
            winners.append(name)
        except:
            print("Game crashed")
            winners.append("Crash")

    makedirs(f"results", exist_ok=True)
    with open(f"results/mcts_results_{mcts_config}.txt", "w") as f:
        for winner in winners:
            f.write(winner + "\n")
