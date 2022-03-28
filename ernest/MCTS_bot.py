import botbowl
from botbowl.core import Action, Agent
import numpy as np
from copy import deepcopy
import random
import time

PRINT = False
IGNORE_IN_GAME = [botbowl.ActionType.PLACE_PLAYER, botbowl.ActionType.END_SETUP, botbowl.ActionType.SETUP_FORMATION_SPREAD, botbowl.ActionType.SETUP_FORMATION_LINE, botbowl.ActionType.SETUP_FORMATION_WEDGE, botbowl.ActionType.SETUP_FORMATION_ZONE]

class Node:
    def __init__(self, action=None, parent=None, C=2):
        self.parent = parent
        self.children = []
        self.action = action
        self.evaluations = []

        self.C = C

        self.n_wins = 0
        self.n_sims = 0


    def UTC(self, root):
        if self.n_sims != 0:
            return self.n_wins / self.n_sims + self.C * (np.sqrt(np.log(root.n_sims) / self.n_sims))
        else:
            return float('inf')


    def extract_children(self, game: botbowl.Game):
        for action_choice in game.get_available_actions():
            # print(action_choice)
            if action_choice.action_type in IGNORE_IN_GAME:
                continue

            for player in action_choice.players:
                self.children.append(Node(Action(action_choice.action_type, player=player), parent=self))
            for position in action_choice.positions:
                self.children.append(Node(Action(action_choice.action_type, position=position), parent=self))
            if len(action_choice.players) == len(action_choice.positions) == 0:
                self.children.append(Node(Action(action_choice.action_type), parent=self))
        return self

class SearchBot(botbowl.Agent):
    def __init__(self, name, budget=1000, time_budget=5, seed=None):
        super().__init__(name)
        self.my_team = None
        self.budget = budget
        self.time_budget = time_budget
        self.path = []

    def new_game(self, game, team):
        self.my_team = team

    def end_game(self, game):
        pass

    def selection(self, node: Node) -> Node:
        return node.children[np.argmax([n.UTC(node) for n in node.children])]

    def rollout(self, game: botbowl.game.Game, node: Node):
        step_before_rollout = game.get_step()
        if PRINT:
            print(f'condition 1: {not game.state.game_over and len(node.children) == 0}')        
        while not game.state.game_over and len(node.children) == 0:
            action = np.random.choice(node.extract_children(game).children).action
            if PRINT:
                print('---------------->', action)
            
            game.step(action)

        win = game.get_winner()
        if PRINT:
            print(f'winner: {win}')
        if win == None:
            # DRAW -- score is zero
            score = -1 
        elif win == self:
            score = 10
        else:
            score = -5  

        game.revert(step_before_rollout) # not sure if necessary

        return score
    

    def expand(self, game: botbowl.Game, node: Node):
        game.step(node.action)
        self.path.append(node)
        node.extract_children(game=game)
        


    def backpropagate(self, score, node: Node):
        for n in range(len(self.path)):
            self.path[n].n_sims += 1
            self.path[n].n_wins += score

        node.n_sims += 1
        node.n_wins += score


    def act(self, game: botbowl.Game):
        game_copy = deepcopy(game)
        game_copy.enable_forward_model()
        game_copy.home_agent.human = True
        game_copy.away_agent.human = True
        root_step = game_copy.get_step()
        root_node = Node()

        available_actions = [elem.action_type for elem in game_copy.get_available_actions()]
        if True:
            print(available_actions)
        # input()

        # if we only have one action, return it, no need to choose what the best action can be
        # if len(available_actions) == 1:
        #     return Action(available_actions[0])

        # handle placing ball randomly on board
        if len(available_actions) == 1:
            if available_actions[0] == botbowl.ActionType.PLACE_BALL:
                if PRINT:
                    print(f'positions: {game_copy.get_available_actions()[0].positions}')
                player_pos = [player.position.x for player in game.get_players_on_pitch(team=self.my_team)]
                if game.is_home_team(team=self.my_team):
                    return Action(botbowl.ActionType.PLACE_BALL, position=botbowl.Square(random.randint(1,8), random.randint(1,9)))
                else:
                    return Action(botbowl.ActionType.PLACE_BALL, position=botbowl.Square(random.randint(9,16), random.randint(1,9)))
            # else:
            #     print(f'single action is: {available_actions[0]}')
            #     input()

        # handle heads or tail
        if botbowl.ActionType.HEADS in available_actions or botbowl.ActionType.TAILS in available_actions:
            return np.random.choice([Action(botbowl.ActionType.HEADS), Action(botbowl.ActionType.TAILS)])

        # handle kick or receive
        if botbowl.ActionType.KICK in available_actions or botbowl.ActionType.RECEIVE in available_actions:
            # return np.random.choice([Action(botbowl.ActionType.KICK), Action(botbowl.ActionType.RECEIVE)])
            return Action(botbowl.ActionType.KICK) # TODO remove 

        # handle the action to setup the bot team
        if botbowl.ActionType.PLACE_PLAYER in available_actions or botbowl.ActionType.END_SETUP in available_actions or botbowl.ActionType.SETUP_FORMATION_SPREAD in available_actions or botbowl.ActionType.SETUP_FORMATION_WEDGE in available_actions:
            available_actions.remove(botbowl.ActionType.PLACE_PLAYER)
            for elem in game_copy.get_players_on_pitch(team=self.my_team):
                return Action(botbowl.ActionType.END_SETUP)
            available_actions.remove(botbowl.ActionType.END_SETUP)
            return Action(np.random.choice(available_actions))

        root_node.extract_children(game=game_copy)
        start = time.time()

        # for i in range(self.budget):
        while time.time() - start < self.time_budget:
            # selection of node
            node = self.selection(root_node)
            self.path = [root_node]

            while True:
                if node.n_sims == 0:
                    score = self.rollout(game=game_copy, node=node)
                    self.backpropagate(score=score, node=node)
                    break
                else:
                    self.expand(game=game_copy, node=node)
                    node = self.selection(node)

                if time.time() - start >= self.time_budget:
                    break
            
            game_copy.revert(root_step)

        return root_node.children[np.argmax([n.n_wins for n in root_node.children])].action

            
            
            # for action_choice in game_copy.get_available_actions():
            #     for player in action_choice.players:
            #         root_node.children.append(Node(Action(action_type=action_choice.action_type, player=player), parent=root_node))
            #     for position in action_choice.positions:
            #         root_node.children.append(Node(Action(action_type=action_choice.action_type, position=position), parent=root_node))
            #     if len(action_choice.players) == len(action_choice.positions) == 0:
            #         print(f'action with no player and no position : {action_choice}')
            



        # for action_choice in game_copy.get_available_actions():
        #     print(f'action_choice: {action_choice}')
        #     if action_choice.action_type == botbowl.ActionType.PLACE_PLAYER:
        #         continue

        #     for player in action_choice.players:
        #         root_node.children.append(Node(Action(action_choice.action_type, player=player), parent=root_node))
        #     for position in action_choice.positions:
        #         root_node.children.append(Node(Action(action_choice.action_type, position=position), parent=root_node))
        #     if len(action_choice.players) == len(action_choice.positions) == 0:
        #         root_node.children.append(Node(Action(action_choice.action_type), parent=root_node))

        # # print(f'actions after : {root_node.children}')


        # best_node = None
        # # print(f"Evaluating {len(root_node.children)} nodes")
        # t = time.time()
        # # for _ in range(self.budget):


        # for node in root_node.children:
        #     game_copy.step(node.action)
        #     while not game.state.game_over and len(game.state.available_actions) == 0:
        #         game_copy.step()
        #     score = self._evaluate(game)
        #     node.visit(score)
        #     print(f"{node.action.action_type}: {node.score()}")
        #     if best_node is None or node.score() > best_node.score():
        #         best_node = node

        #     game_copy.revert(root_step)

        # # print(f"{best_node.action.action_type} selected in {time.time() - t} seconds")
        # print(f"best action : {best_node.action}")


# Register the bot to the framework
botbowl.register_bot('MCTS-bot', SearchBot)

if __name__ == '__main__':
    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl")
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False
    config.fast_mode = True
    config.pathfinding_enabled = False

    # Play a game
    bot_a = botbowl.make_bot("search-bot")
    bot_b = botbowl.make_bot("search-bot")
    game = botbowl.Game(1, home, away, bot_a, bot_b, config, arena=arena, ruleset=ruleset)
    print("Starting game")
    game.init()
    print(game.get_winner())
    print("Game is over")
