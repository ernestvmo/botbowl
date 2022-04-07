import botbowl
import numpy as np
from copy import deepcopy
import time

class Node:
    def __init__(self, action=None, parent=None, C=2):
        """Initializer method for the Node class.

        Args:
            action (botbowl.Action, optional): Game action that lead to this state. Defaults to None.
            parent (Node, optional): Parent node of this node object. Defaults to None.
            C (int, optional): C parameter to calculate the UTC scores of a node. Defaults to 2.
        """
        self.parent = parent
        self.children = []
        self.action = action
        self.evaluations = []

        self.C = C

        self.n_wins = 0
        self.n_sims = 0

    def UTC(self, root) -> float:
        """ Calculate the UTC score of this node, using the passed parameter as root node of the tree.

        Args:
            root (Node): The root Node of the tree.

        Returns:
            float: if the node has been visited, calculate the UTC score of that node, otherwise return 'inf'.
        """
        if self.n_sims != 0:
            return self.n_wins / self.n_sims + self.C * (np.sqrt(np.log(root.n_sims) / self.n_sims))
        else:
            return float('inf')


    def extract_children(self, game: botbowl.Game):
        """Extract all possible actions at the given game state and set them as children of the current Node object.
        All possible actions at the current game state is appended to the Node's children list.

        Args:
            game (botbowl.Game): The current game state.

        Returns:
            Node: Return the Node object with its children (all possible actions at the current game state) extracted.
        """
        for action_choice in game.get_available_actions():
            for player in action_choice.players:
                self.children.append(Node(botbowl.Action(action_choice.action_type, player=player), parent=self))
            for position in action_choice.positions:
                self.children.append(Node(botbowl.Action(action_choice.action_type, position=position), parent=self))
            if len(action_choice.players) == len(action_choice.positions) == 0:
                self.children.append(Node(botbowl.Action(action_choice.action_type), parent=self))
        return self

class SearchBot(botbowl.Agent):
    '''Bot agent for Monte Carlo Tree search in the botbowl framework.'''
    def __init__(self, name: str, budget=200, time_budget=5):
        """Initializer method for the SearchBot class.

        Args:
            name (str): The bot name.
            budget (int, optional): Epoch budget for the Monte Carlo Tree Search rollout operation. Defaults to 200.
            time_budget (int, optional): Time budget (in seconds) for the Monte Carlo Tree Search (unused in current implementation). Defaults to 5.
        """
        super().__init__(name)
        self.my_team = None

        self.budget = budget
        self.time_budget = time_budget

        # list of actions taken
        self.path = []

        # last action taken by the bot
        self.last_action = None
        
        # used for debugging errors
        self.debug = False

    def new_game(self, game: botbowl.Game, team: botbowl.Team):
        '''Creates a new game, assigning the passed team parameter as the team operated by the bot.'''
        self.my_team = team

    def end_game(self, game: botbowl.Game):
        '''Declares the end of a game.'''
        game._end_game()

    def selection(self, node: Node) -> Node:
        """Selection operation of the Monte Carlo Tree Search. Using argmax, returns the Node with the highest UTC score of the passed Node parameter.

        Args:
            node (Node): The Node object we want to select the children of.

        Returns:
            Node: The node object with the highest UTC score out of all the available children.
        """
        return node.children[np.argmax([n.UTC(node) for n in node.children])]

    def rollout(self, game: botbowl.Game, node: Node):
        """Rollout operation of the Monte Carlo Tree Search.
        During rollout, we execute random actions from the available actions until we reach a terminal state.
        Once at that state, we evaluate the use the terminated state to determine winner of the game.\n
        If the winner is equals to None, this means the game ended in a DRAW, we return a score of -1. If the winner equals to the team of the bot, the bot has won and we return a score of 10. If the winner of the game is neither None nor the bot's team, this means that the bot lost and we return a score of -5.

        Args:
            game (botbowl.Game): A copy of the game to use for rollout.
            node (Node): The node that was selected to start the rollout operation from.

        Returns:
            int: The score reached at the terminal state.
        """
        step_before_rollout = game.get_step()
        if self.debug:
            print(f'condition 1: {not game.state.game_over and len(node.children) == 0}')
        while not game.state.game_over and len(node.children) == 0:
            action = np.random.choice(node.extract_children(game).children).action
            if action.action_type != botbowl.ActionType.PLACE_PLAYER:
                # this action might appear as availble if a player was downed, 
                # but we do not want to take it as it would break the game
                game.step(action)

        # retrieve winner of the game
        win = game.get_winner()
        if self.debug:
            print(f'winner: {win}')
        if win == None:
            # DRAW  -- no one won
            score = -1 
        elif win == self:
            # WIN   -- the bot won
            score = 10
        else:
            # LOST  -- the bot lost
            score = -5  

        # rollback the game state before we did the rollout
        game.revert(step_before_rollout) # not sure if necessary

        return score
    

    def expand(self, game: botbowl.Game, node: Node):
        """Expand the provided node by adding it to the list of traversed nodes, then extract the children of that node.

        Args:
            game (botbowl.Game): The current game state.
            node (Node): The node that we traversed and extract the children of.
        """
        game.step(node.action)
        self.path.append(node)
        node.extract_children(game=game)
        

    def backpropagate(self, score: int, node: Node):
        """Update the number of simulations  and number of wins for the passed node and for all nodes in the list of traversed nodes.

        Args:
            score (int): The score obtained from the rollout operation.
            node (Node): The last visited node.
        """
        for n in range(len(self.path)):
            self.path[n].n_sims += 1
            self.path[n].n_wins += score

        node.n_sims += 1
        node.n_wins += score


    def act(self, game: botbowl.Game) -> botbowl.Action:
        """Use Monte Carlo Tree Search to find the best action to take at given game state.

        Args:
            game (botbowl.Game): The current game state.

        Returns:
            botbowl.Action: The best action to take per dedided by the Monte Carlo Tree Search.
        """
        # make a copy of the game before we take any action, so we can revert to it afterwards
        game_copy = deepcopy(game)
        game_copy.enable_forward_model()
        game_copy.home_agent.human = True
        game_copy.away_agent.human = True
        root_step = game_copy.get_step()

        # define root node
        root_node = Node()
        # extract all possible actions at the given game state
        available_actions = [elem.action_type for elem in game_copy.get_available_actions()]
        
        if self.debug:
            print(available_actions)

        # some actions are simple and do not require the use of Monte Carlo Tree Search, or would be faster without it
        # handle placing ball randomly on board
        if len(available_actions) == 1:
            if available_actions[0] == botbowl.ActionType.PLACE_BALL:
                if self.debug:
                    print(f'positions: {game_copy.get_available_actions()[0].positions}')
                return botbowl.Action(botbowl.ActionType.PLACE_BALL, position=np.random.choice(game.get_available_actions()[0].positions))

        # handle heads or tail
        if botbowl.ActionType.HEADS in available_actions or botbowl.ActionType.TAILS in available_actions:
            return np.random.choice([botbowl.Action(botbowl.ActionType.HEADS), botbowl.Action(botbowl.ActionType.TAILS)])

        # handle kick or receive
        if botbowl.ActionType.KICK in available_actions or botbowl.ActionType.RECEIVE in available_actions:
            # return np.random.choice([botbowl.Action(botbowl.ActionType.KICK), botbowl.Action(botbowl.ActionType.RECEIVE)])
            return botbowl.Action(botbowl.ActionType.KICK) # TODO remove 

        # handle the action to setup the bot team
        if botbowl.ActionType.PLACE_PLAYER in available_actions or botbowl.ActionType.END_SETUP in available_actions or botbowl.ActionType.SETUP_FORMATION_SPREAD in available_actions or botbowl.ActionType.SETUP_FORMATION_WEDGE in available_actions:
            available_actions.remove(botbowl.ActionType.PLACE_PLAYER)
            for elem in game_copy.get_players_on_pitch(team=self.my_team):
                return botbowl.Action(botbowl.ActionType.END_SETUP)
            available_actions.remove(botbowl.ActionType.END_SETUP)
            return botbowl.Action(np.random.choice(available_actions))

        # when the ball is on the floor, the bot will attemps to go pick it up
        if game.get_ball().on_ground and botbowl.ActionType.MOVE in available_actions and self.last_action == botbowl.ActionType.START_MOVE:
            return botbowl.Action(botbowl.ActionType.MOVE, game.get_ball().position, \
                player=np.random.choice(game.get_players_on_pitch(team=self.my_team)))

        root_node.extract_children(game=game_copy)

        # (unused) timer start for time budget calculations
        start = time.time()

        for _ in range(self.budget):
        # while time.time() - start < self.time_budget:
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

                # if time.time() - start >= self.time_budget:
                #     break
            
            game_copy.revert(root_step)

        # select the best action from the root node's children using argmax, set it as the last action
        self.last_action = root_node.children[np.argmax([n.n_wins for n in root_node.children])].action
        return self.last_action


# Register the bot to the framework
botbowl.register_bot('MCTS-bot', SearchBot)

# used only when testing MCTS vs MCTS
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
    bot_a = botbowl.make_bot("MCTS-bot")
    bot_b = botbowl.make_bot("MCTS-bot")
    game = botbowl.Game(1, home, away, bot_a, bot_b, config, arena=arena, ruleset=ruleset)
    print("Starting game")
    game.init()
    print('Winner is:', game.get_winner())
    print("Game is over")
