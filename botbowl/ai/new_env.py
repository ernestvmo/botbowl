"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains the botbowlEnv class; botbowl implementing the Open AI Gym interface.
"""
from typing import Tuple, Iterable, Union, Callable

from botbowl.ai.bots import RandomBot
from botbowl.ai.registry import registry as bot_registry
from botbowl.ai.layers import *
from botbowl.core import Game, load_rule_set, load_config, load_team_by_filename, load_arena, load_formation

import gym
import uuid
from itertools import count
from copy import deepcopy


def take(n: int, iterable: Iterable) -> None:
    for _ in range(n):
        next(iterable)


class EnvConf:
    config: Configuration = load_config("gym-11")
    formations_paths = [
        'def_spread.txt',
        'def_zone.txt',
        'off_line.txt',
        'off_wedge.txt'
    ]

    simple_action_types: List[Union[ActionType, Formation]] = [
        ActionType.START_GAME,
        ActionType.HEADS,
        ActionType.TAILS,
        ActionType.KICK,
        ActionType.RECEIVE,
        # ActionType.END_SETUP,
        ActionType.END_PLAYER_TURN,
        ActionType.USE_REROLL,
        ActionType.DONT_USE_REROLL,
        ActionType.USE_SKILL,
        ActionType.DONT_USE_SKILL,
        ActionType.END_TURN,
        ActionType.STAND_UP,
        ActionType.SELECT_ATTACKER_DOWN,
        ActionType.SELECT_BOTH_DOWN,
        ActionType.SELECT_PUSH,
        ActionType.SELECT_DEFENDER_STUMBLES,
        ActionType.SELECT_DEFENDER_DOWN,
        ActionType.SELECT_NONE,
        ActionType.USE_BRIBE,
        ActionType.DONT_USE_BRIBE,
    ]
    formations = [load_formation(formation, size=11) for formation in formations_paths]

    simple_action_types.extend(formations)

    positional_action_types = [
        ActionType.PLACE_BALL,
        ActionType.PUSH,
        ActionType.FOLLOW_UP,
        ActionType.MOVE,
        ActionType.BLOCK,
        ActionType.PASS,
        ActionType.FOUL,
        ActionType.HANDOFF,
        ActionType.LEAP,
        ActionType.STAB,
        ActionType.SELECT_PLAYER,
        ActionType.START_MOVE,
        ActionType.START_BLOCK,
        ActionType.START_BLITZ,
        ActionType.START_PASS,
        ActionType.START_FOUL,
        ActionType.START_HANDOFF
    ]

    action_types = simple_action_types + positional_action_types

    layers: List[FeatureLayer] = [
        OccupiedLayer(),
        OwnPlayerLayer(),
        OppPlayerLayer(),
        OwnTackleZoneLayer(),
        OppTackleZoneLayer(),
        UpLayer(),
        StunnedLayer(),
        UsedLayer(),
        RollProbabilityLayer(),
        BlockDiceLayer(),
        ActivePlayerLayer(),
        TargetPlayerLayer(),
        MALayer(),
        STLayer(),
        AGLayer(),
        AVLayer(),
        MovementLeftLayer(),
        GFIsLeftLayer(),
        BallLayer(),
        OwnHalfLayer(),
        OwnTouchdownLayer(),
        OppTouchdownLayer(),
        SkillLayer(Skill.BLOCK),
        SkillLayer(Skill.DODGE),
        SkillLayer(Skill.SURE_HANDS),
        SkillLayer(Skill.CATCH),
        SkillLayer(Skill.PASS)
    ]
    layers.extend(AvailablePositionLayer(action_type) for action_type in positional_action_types)

    # Procedures that require actions
    procedures: List[Procedure] = [
        StartGame,
        CoinTossFlip,
        CoinTossKickReceive,
        Setup,
        PlaceBall,
        HighKick,
        Touchback,
        Turn,
        MoveAction,
        BlockAction,
        BlitzAction,
        PassAction,
        HandoffAction,
        FoulAction,
        ThrowBombAction,
        Block,
        Push,
        FollowUp,
        Apothecary,
        PassAttempt,
        Interception,
        Reroll,
        Ejection]


class NewBotBowlEnv(gym.Env):
    """
    Environment for Bot Bowl IV targeted at reinforcement learning algorithms
    """

    layers: FeatureLayer
    width: int
    height: int
    board_squares: int
    num_actions: int
    game: Game
    _seed: int
    rnd: np.random.RandomState
    ruleset: RuleSet
    home_team: Team
    away_team: Team
    num_non_spatial_observables: int

    def __init__(self, seed: int = None, home_agent='human', away_agent='random'):

        # Game
        self.game = None
        self.ruleset = load_rule_set(EnvConf.config.ruleset, all_rules=False)
        self.home_team = load_team_by_filename('human', self.ruleset, board_size=11)
        self.away_team = load_team_by_filename('human', self.ruleset, board_size=11)
        self.home_agent = home_agent
        self.away_agent = away_agent

        arena = load_arena(EnvConf.config.arena)
        self.width = arena.width
        self.height = arena.height
        self.board_squares = self.width * self.height
        self.num_non_spatial_observables = None

        # Gym stuff
        self._seed = np.random.randint(0, 2 ** 31) if seed is None else seed
        self.rnd = np.random.RandomState(self._seed)

        # Setup gym shapes
        spat_obs = self.reset()
        self.action_space = gym.spaces.Discrete(len(EnvConf.action_types))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=spat_obs.shape)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: tuple with np arrays
                     spatial observation with shape=(num_layers, height, width)
                     non spatial observation with shape=(num_non_spatial_observations)
                     action mask with shape = (action_space, )"""

        game = self.game
        active_team = game.active_team
        active_player = game.state.active_player
        opp_team = game.get_opp_team(active_team) if active_team is not None else None

        # Spatial state
        spatial_obs = np.stack([layer.get(game) for layer in EnvConf.layers])
        if self._flip_x_axis():
            spatial_obs = np.flip(spatial_obs, axis=2)

        # Non spatial state
        if self.num_non_spatial_observables is None:
            non_spatial_obs = np.zeros(2000)
        else:
            non_spatial_obs = np.zeros(self.num_non_spatial_observables)

        index = count()
        non_spatial_obs[next(index)] = game.state.half - 1.0
        non_spatial_obs[next(index)] = game.state.round / 8.0

        non_spatial_obs[next(index)] = 1.0 * (game.state.weather == WeatherType.SWELTERING_HEAT)
        non_spatial_obs[next(index)] = 1.0 * (game.state.weather == WeatherType.VERY_SUNNY)
        non_spatial_obs[next(index)] = 1.0 * (game.state.weather == WeatherType.NICE)
        non_spatial_obs[next(index)] = 1.0 * (game.state.weather == WeatherType.POURING_RAIN)
        non_spatial_obs[next(index)] = 1.0 * (game.state.weather == WeatherType.BLIZZARD)

        non_spatial_obs[next(index)] = 1.0 * (game.state.current_team == active_team)
        non_spatial_obs[next(index)] = 1.0 * (game.state.kicking_first_half == active_team)
        non_spatial_obs[next(index)] = 1.0 * (game.state.kicking_this_drive == active_team)
        non_spatial_obs[next(index)] = len(game.get_reserves(active_team)) / 16.0
        non_spatial_obs[next(index)] = len(game.get_knocked_out(active_team)) / 16.0
        non_spatial_obs[next(index)] = len(game.get_casualties(active_team)) / 16.0
        non_spatial_obs[next(index)] = len(game.get_reserves(game.get_opp_team(active_team))) / 16.0
        non_spatial_obs[next(index)] = len(game.get_knocked_out(game.get_opp_team(active_team))) / 16.0
        non_spatial_obs[next(index)] = len(game.get_casualties(game.get_opp_team(active_team))) / 16.0

        if active_team is not None:
            non_spatial_obs[next(index)] = active_team.state.score / 16.0
            non_spatial_obs[next(index)] = active_team.state.turn / 8.0
            non_spatial_obs[next(index)] = active_team.state.rerolls_start / 8.0
            non_spatial_obs[next(index)] = active_team.state.rerolls / 8.0
            non_spatial_obs[next(index)] = active_team.state.ass_coaches / 8.0
            non_spatial_obs[next(index)] = active_team.state.cheerleaders / 8.0
            non_spatial_obs[next(index)] = active_team.state.bribes / 4.0
            non_spatial_obs[next(index)] = active_team.state.babes / 4.0
            non_spatial_obs[next(index)] = active_team.state.apothecaries / 2
            non_spatial_obs[next(index)] = 1.0 * (not active_team.state.reroll_used)
            non_spatial_obs[next(index)] = active_team.state.fame / 2

            non_spatial_obs[next(index)] = opp_team.state.score / 16.0
            non_spatial_obs[next(index)] = opp_team.state.turn / 8.0
            non_spatial_obs[next(index)] = opp_team.state.rerolls_start / 8.0
            non_spatial_obs[next(index)] = opp_team.state.rerolls / 8.0
            non_spatial_obs[next(index)] = opp_team.state.ass_coaches / 8.0
            non_spatial_obs[next(index)] = opp_team.state.cheerleaders / 8.0
            non_spatial_obs[next(index)] = opp_team.state.bribes / 4.0
            non_spatial_obs[next(index)] = opp_team.state.babes / 4.0
            non_spatial_obs[next(index)] = active_team.state.apothecaries / 2
            non_spatial_obs[next(index)] = 1.0 * (not opp_team.state.reroll_used)
            non_spatial_obs[next(index)] = opp_team.state.fame / 2
        else:
            take(22, index)

        if game.current_turn() is not None:
            non_spatial_obs[next(index)] = 1.0 * game.is_blitz_available()
            non_spatial_obs[next(index)] = 1.0 * game.is_pass_available()
            non_spatial_obs[next(index)] = 1.0 * game.is_handoff_available()
            non_spatial_obs[next(index)] = 1.0 * game.is_foul_available()
            non_spatial_obs[next(index)] = 1.0 * game.is_blitz()
            non_spatial_obs[next(index)] = 1.0 * game.is_quick_snap()
        else:
            take(6, index)

        if active_player is not None:
            player_action_type = game.get_player_action_type()
            non_spatial_obs[next(index)] = 1.0 * (player_action_type == PlayerActionType.MOVE)
            non_spatial_obs[next(index)] = 1.0 * (player_action_type == PlayerActionType.BLOCK)
            non_spatial_obs[next(index)] = 1.0 * (player_action_type == PlayerActionType.BLITZ)
            non_spatial_obs[next(index)] = 1.0 * (player_action_type == PlayerActionType.PASS)
            non_spatial_obs[next(index)] = 1.0 * (player_action_type == PlayerActionType.HANDOFF)
            non_spatial_obs[next(index)] = 1.0 * (player_action_type == PlayerActionType.FOUL)
        else:
            take(6, index)

        # Procedures in stack
        all_proc_types = set(type(proc) for proc in game.state.stack.items)
        for i, proc_type in zip(index, EnvConf.procedures):
            non_spatial_obs[i] = 1.0 * (proc_type in all_proc_types)

        # Available action types
        aa_types = np.zeros(len(EnvConf.action_types))
        game_aa_types = set(action_choice.action_type for action_choice in game.get_available_actions())
        is_setup: bool = type(self.game.get_procedure()) == Setup
        for i, action_type in enumerate(EnvConf.action_types):
            if action_type is ActionType.END_SETUP and not game.is_setup_legal(active_team):
                continue  # Ignore end setup action if setup is illegal

            if is_setup and isinstance(action_type, Formation):
                aa_types[i] = 1.0
            else:
                aa_types[i] = 1.0 * (action_type in game_aa_types)

        next_index = next(index)
        non_spatial_obs[next_index:next_index + len(aa_types)] = aa_types
        if self.num_non_spatial_observables is None:
            self.num_non_spatial_observables = next_index + len(aa_types)
            non_spatial_obs = non_spatial_obs[:self.num_non_spatial_observables]

        # Action mask
        num_simple_actions = len(EnvConf.simple_action_types)
        aa_layer_first_index = len(EnvConf.layers) - len(EnvConf.positional_action_types)
        action_mask = np.concatenate((aa_types[:num_simple_actions],
                                      spatial_obs[aa_layer_first_index:].flatten()))
        assert 1.0 in action_mask

        return spatial_obs, non_spatial_obs, action_mask

    def step(self, action: int, skip_observation: bool = False):
        # Convert to Action object
        action_objects = self._compute_action(action, flip=self._flip_x_axis())
        active_team = self.game.active_team

        for action in action_objects:
            self.game.step(action)

        reward = 0
        done = self.game.state.game_over
        if done:
            if self.game.get_winning_team() is active_team:
                reward = 1
            elif self.game.get_winning_team() is self.game.get_opp_team(active_team):
                reward = -1

        if done or skip_observation:
            spatial_observation, non_spatial_observation, action_mask = None, None, None
        else:
            spatial_observation, non_spatial_observation, action_mask = self.get_state()

        info = {'non_spatial_obs': non_spatial_observation,
                'action_mask': action_mask}

        return spatial_observation, reward, done, info

    def render(self, mode='human'):
        pass

    def reset(self):
        seed = self.rnd.randint(0, 2 ** 31)

        self.game = Game(game_id=str(uuid.uuid1()),
                         home_team=deepcopy(self.home_team),
                         away_team=deepcopy(self.away_team),
                         home_agent=NewBotBowlEnv._create_agent(self.home_agent),
                         away_agent=NewBotBowlEnv._create_agent(self.home_agent),
                         config=EnvConf.config,
                         ruleset=self.ruleset,
                         seed=seed)

        self.game.init()
        return self.get_state()[0]

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            self._seed = seed
            self.rnd = np.random.RandomState(self._seed)
        return self._seed

    def _flip_x_axis(self) -> bool:
        """Returns true if x-axis should be flipped"""
        return self.game.active_team is self.game.state.away_team

    def _compute_action(self, action_idx: int, flip: bool) -> List[Action]:
        if action_idx < len(EnvConf.simple_action_types):
            if action_idx >= len(EnvConf.simple_action_types) - len(EnvConf.formations):
                formation = EnvConf.simple_action_types[action_idx]
                return formation.actions(self.game, self.game.active_team) + [Action(ActionType.END_SETUP)]
            else:
                return [Action(EnvConf.simple_action_types[action_idx])]

        spatial_idx = action_idx - len(EnvConf.simple_action_types)
        spatial_pos_idx = spatial_idx % self.board_squares
        spatial_y = int(spatial_pos_idx // self.width)
        spatial_x = int(spatial_pos_idx % self.width)
        if flip:
            spatial_x = self.width - spatial_x - 1

        spatial_action_type = EnvConf.positional_action_types[spatial_idx // self.board_squares]
        return [Action(spatial_action_type, self.game.get_square(spatial_x, spatial_y))]

    def _compute_action_idx(self, action: Action) -> int:
        if action.action_type in EnvConf.simple_action_types:
            return EnvConf.simple_action_types.index(action.action_type)
        elif action.action_type in EnvConf.positional_action_types:
            position = action.position if action.position is not None else action.player.position
            spatial_index = position.x + position.y * self.width
            position_action_index = EnvConf.positional_action_types.index(action.action_type)
            return len(EnvConf.simple_action_types) + self.board_squares * position_action_index + spatial_index
        else:
            raise AttributeError(f"Can't convert {action} to an action index")

    @staticmethod
    def _create_agent(agent_option) -> Agent:
        if agent_option == "human":
            return Agent("Gym Learner", human=True)
        elif agent_option == "random":
            return RandomBot("Random bot")
        elif agent_option in bot_registry.list():
            return bot_registry.make(agent_option)
        else:
            raise AttributeError(f"Not regonized bot name: {agent_option}")


class BotBowlWrapper:
    env: NewBotBowlEnv

    def __init__(self, env):
        self.env = env

    def get_state(self):
        return self.env.get_state()

    def step(self, action: int, skip_observation: bool = False):
        return self.env.step(action, skip_observation)

    def render(self, mode='human'):
        return self.env.render(mode)

    def reset(self):
        return self.env.reset()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


class RewardWrapper(BotBowlWrapper):
    reward_func: Callable[[Game], float]

    def __init__(self, env, reward_func: Callable[[Game], float]):
        super().__init__(env)
        self.home_accumulated_reward = 0
        self.away_accumulated_reward = 0
        self.reward_func = reward_func
        self.index_last_report = 0

    def step(self, action: int, skip_observation: bool = False):
        self.env.step(action, skip_observation)

    def reset(self):
        self.env.reset()
        self.home_accumulated_reward = 0
        self.away_accumulated_reward = 0
        self.index_last_report = 0


class ScriptedActionWrapper(BotBowlWrapper):
    def __init__(self):
        pass