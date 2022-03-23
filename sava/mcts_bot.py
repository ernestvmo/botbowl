#!/usr/bin/env python3

from copy import deepcopy
from typing import List, Tuple
from typing_extensions import Self
import botbowl
import numpy as np
from botbowl.ai.bots.random_bot import RandomBot

from botbowl.core.game import Game
from botbowl.core.model import Action, ActionChoice, Agent
from botbowl.core.table import ActionType
from botbowl import Formation, ProcBot
from botbowl.core.procedure import *

class MctsBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)
        self.evals = 0
        self.my_team = None
        self.opp_team = None
        self.actions = []
        self.last_turn = 0
        self.last_half = 0

        self.off_formation = [
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "m", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "S"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x"],
            ["-", "-", "-", "-", "-", "s", "-", "-", "-", "0", "-", "-", "S"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "S"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "m", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
        ]

        self.def_formation = [
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "b", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "S", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "S", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "b", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
        ]

        self.off_formation = Formation("Wedge offense", self.off_formation)
        self.def_formation = Formation("Zone defense", self.def_formation)
        self.setup_actions = []

    def coin_toss_flip(self, game):
        """
        Select heads/tails and/or kick/receive
        """
        return Action(ActionType.TAILS)
        # return Action(ActionType.HEADS)

    def coin_toss_kick_receive(self, game):
        """
        Select heads/tails and/or kick/receive
        """
        return Action(ActionType.RECEIVE)
        # return Action(ActionType.KICK)

    def setup(self, game):
        """
        Use either a Wedge offensive formation or zone defensive formation.
        """
        # Update teams
        self.my_team = game.get_team_by_id(self.my_team.team_id)
        self.opp_team = game.get_opp_team(self.my_team)

        if self.setup_actions:
            action = self.setup_actions.pop(0)
            return action
        else:
            if game.get_receiving_team() == self.my_team:
                self.setup_actions = self.off_formation.actions(game, self.my_team)
                self.setup_actions.append(Action(ActionType.END_SETUP))
            else:
                self.setup_actions = self.def_formation.actions(game, self.my_team)
                self.setup_actions.append(Action(ActionType.END_SETUP))
            action = self.setup_actions.pop(0)
            return action


    class McNode():
        def __init__(self, bot, game: Game, action: Action, parent: Self, team, random: np.random.RandomState):
            self.bot = bot
            self.game = game
            self.action = action
            self.team = team
            self.rnd = random
            
            self.expanded = False
            self.simulated = False
            self.result = None

            self.n_simulations = 0
            self.n_wins = 0
            self.parent = parent

            self.children: List[MctsBot.McNode] = []
        
        def expand(self):
            self.expanded = True
            # Expand this node and set children accordingly.
            for a in self.game.get_available_actions():
                # Create child
                # if a.action_type == ActionType.PLACE_PLAYER:
                #     continue
                
                if a.players == None:
                    pass

                for player in a.players:
                    # sampled_pos = self.rnd.choice(, 5)
                    # for pos in sampled_pos:
                    game_copy = deepcopy(self.game)
                    action_taken = botbowl.Action(a.action_type, player=player)
                    game_copy._one_step(action_taken)
                    # Flip step of other bot
                    c = MctsBot.McNode(self.bot, game_copy, action_taken,self, self.team, self.rnd)
                    c.playout()
                    self.children.append(c)

        def UCT(self) -> float:
            # TODO C param
            return (self.n_wins / self.n_simulations) + (np.sqrt(2) * np.sqrt(np.log(self.parent.n_simulations) / self.n_simulations)) 

        def find_best_child(self) -> Tuple[Self, bool]:
            if self.expanded:
                return self.children[np.argmax([n.UCT() for n in self.children])], True
            return self, False

        def playout(self):
            self.bot.evals += 1
            self.simulated = True
            self.n_simulations += 1

            # Randomly play out from this node onwards.
            # and store result in node.
            game_copy = deepcopy(self.game)
            while (not game_copy.state.game_over) and len(game_copy.get_available_actions()) != 0:
                a = self.select_random_action(game_copy)
                game_copy._one_step(a)

            winner =  self.game.get_winning_team()
            if winner == self.team:
                self.result = 1
            elif winner is None:
                self.result = 0
            else:
                # Other team won
                self.result = -1

        def select_random_action(self, game):
            # Select a random action type
            while True:
                action_choice = self.rnd.choice(game.get_available_actions())
                # Ignore PLACE_PLAYER actions
                if action_choice.action_type != botbowl.ActionType.PLACE_PLAYER:
                    break

            # Select a random position and/or player
            position = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
            player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None

            # Make action object
            action = botbowl.Action(action_choice.action_type, position=position, player=player)

            # Return action to the framework
            return action


    def new_game(self, game, team):
        self.my_team = team

    def act(self, game):
        proc = game.get_procedure()

        if isinstance(proc, CoinTossFlip):
            return self.coin_toss_flip(game)
        elif isinstance(proc, CoinTossKickReceive):
            return self.coin_toss_kick_receive(game)
        elif isinstance(proc, Setup):
            # if proc.reorganize:
            #     action = self.perfect_defense(game)

                return self.setup(game)

    
        root = MctsBot.McNode(self, game, None, None, self.my_team, self.rnd)
        while self.evals < 100:
            c, has_children = root.find_best_child()
            path = [c]
            # Fully explore tree
            while has_children:
                c, has_children = c.find_best_child()
                path.append(c)

            c.expand()
            c.playout()
            # Propogate the path futher
            for p in path:
                p.n_simulations += 1
                if c.result == 1:
                    p.n_simulations += 1

        best, _ = root.find_best_child()
        return best.action
            


    def end_game(self, game):
        pass


# Register the bot to the framework
botbowl.register_bot('sava-mcts', MctsBot)
botbowl.register_bot('random-bot', RandomBot)

if __name__ == "__main__":

    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = False
    
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    # Play 10 games
    game_times = []
    for i in range(10):
        away_agent = botbowl.make_bot("sava-mcts")
        home_agent = botbowl.make_bot("random-bot")

        game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.enable_forward_model()
        game.config.fast_mode = True

        print("Starting game", (i+1))
        game.init()
        print("Game is over")
