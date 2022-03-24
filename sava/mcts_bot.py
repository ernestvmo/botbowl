#!/usr/bin/env python3

from copy import deepcopy
from typing import List, Tuple
from typing_extensions import Self
import botbowl
import numpy as np
from botbowl.ai.bots.random_bot import RandomBot

from botbowl.core.game import Game
from botbowl import Action, ActionChoice, Square
from botbowl.core.table import ActionType


class MctsBot(botbowl.Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.opp_team = None
        self.rnd = np.random.RandomState(seed)
        self.evals = 0

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
                if a.players:
                    for player in a.players:
                        if a.positions:
                            for pos in a.positions:
                                    game_copy = deepcopy(self.game)
                                    action_taken = botbowl.Action(a.action_type,position=pos,  player=player)
                                    game_copy._one_step(action_taken)
                                    # Flip step of other bot
                                    c = MctsBot.McNode(self.bot, game_copy, action_taken,self, self.team, self.rnd)
                                    c.playout()
                                    self.children.append(c)
                        else:
                            game_copy = deepcopy(self.game)
                            action_taken = botbowl.Action(a.action_type,  player=player)
                            game_copy._one_step(action_taken)
                            # Flip step of other bot
                            c = MctsBot.McNode(self.bot, game_copy, action_taken,self, self.team, self.rnd)
                            c.playout()
                            self.children.append(c)
                else:
                    if a.positions:
                        for pos in a.positions:
                                game_copy = deepcopy(self.game)
                                action_taken = botbowl.Action(a.action_type,position=pos)
                                game_copy.step(action_taken)
                                # Flip step of other bot
                                c = MctsBot.McNode(self.bot, game_copy, action_taken,self, self.team, self.rnd)
                                c.playout()
                                self.children.append(c)
                    game_copy = deepcopy(self.game)
                    action_taken = botbowl.Action(a.action_type)
                    game_copy._one_step(action_taken)
                    # Flip step of other bot
                    c = MctsBot.McNode(self.bot, game_copy, action_taken,self, self.team, self.rnd)
                    c.playout()
                    self.children.append(c)

        def UCT(self) -> float:
            # TODO C param
            return (self.n_wins / self.n_simulations) + (0.1 * np.sqrt(np.log(self.parent.n_simulations) / self.n_simulations)) 

        def find_best_child(self) -> Tuple[Self, bool]:
            if self.expanded:
                if self.children:
                    return self.children[np.argmax([n.UCT() for n in self.children])], True
            return self, False

        def playout(self):
            self.bot.evals += 1
            self.simulated = True
            self.n_simulations += 1

            # Randomly play out from this node onwards.
            # and store result in node.
            game_copy = deepcopy(self.game)
            while (not game_copy.state.game_over) and len(game_copy.state.available_actions) != 0:
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
                action_choice = self.rnd.choice(game.state.available_actions)
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
        self.opp_team = game.get_opp_team(team)


    def act(self, game):
        # Build tree
        
        # Select a random action type

        

        available_actions = [elem.action_type for elem in game.get_available_actions()]
        print(available_actions)
        if botbowl.ActionType.PLACE_BALL in available_actions:
            left_center = Square(7, 8)
            right_center = Square(20, 8)
            if game.is_team_side(left_center, self.opp_team):
                return Action(ActionType.PLACE_BALL, position=left_center)
            return Action(ActionType.PLACE_BALL, position=right_center)


        if len(available_actions) == 1:
            return Action(available_actions[0])

        # handle heads or tail
        if botbowl.ActionType.HEADS in available_actions or botbowl.ActionType.TAILS in available_actions:
            return np.random.choice([Action(botbowl.ActionType.HEADS), Action(botbowl.ActionType.TAILS)])

        # handle kick or receive
        if botbowl.ActionType.KICK in available_actions or botbowl.ActionType.RECEIVE in available_actions:
            return np.random.choice([Action(botbowl.ActionType.KICK), Action(botbowl.ActionType.RECEIVE)])

        if botbowl.ActionType.MOVE in available_actions:
             available_actions.remove(botbowl.ActionType.MOVE)


        if botbowl.ActionType.PLACE_PLAYER in available_actions or botbowl.ActionType.END_SETUP in available_actions or botbowl.ActionType.SETUP_FORMATION_SPREAD in available_actions or botbowl.ActionType.SETUP_FORMATION_WEDGE in available_actions:
            available_actions.remove(botbowl.ActionType.PLACE_PLAYER)
            for elem in game.get_players_on_pitch(team=self.my_team):
                return Action(botbowl.ActionType.END_SETUP)
            available_actions.remove(botbowl.ActionType.END_SETUP)
            return Action(np.random.choice(available_actions))
        



            

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
                    p.n_wins += 1

        best, _ = root.find_best_child()
        print(best.action)
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
        game.config.fast_mode = True

        print("Starting game", (i+1))
        game.init()
        print("Game is over")
