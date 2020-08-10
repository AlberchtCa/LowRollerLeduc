import GameFunctions
from typing import List, Dict
from copy import deepcopy
import numpy as np
import random


NUM_ACTIONS = 5


class Player:
    def __init__(self, index: int):
        self.index = index
        self.invested = 1


class InformationSet:
    def __init__(self):
        self.cumulative_regrets = np.zeros(shape=NUM_ACTIONS)
        self.strategy_sum = np.zeros(shape=NUM_ACTIONS)

    @staticmethod
    def normalize(strategy: np.array) -> np.array:
        if sum(strategy) > 0:
            strategy /= sum(strategy)
        else:
            strategy = np.array([1.0 / NUM_ACTIONS] * NUM_ACTIONS)
        return strategy

    def get_strategy(self, reach_probability: float) -> np.array:
        strategy = np.maximum(0, self.cumulative_regrets)
        strategy = self.normalize(strategy)

        self.strategy_sum += reach_probability * strategy
        return strategy

    def get_average_strategy(self) -> np.array:
        return self.normalize(self.strategy_sum.copy())


class GameState:
    def __init__(self, cards):
        self.cards = cards
        self.history = []
        self.players = [Player(0), Player(1)]
        self.acting_player = 0
        self.street = 0
        self.re_raise_param = 0
        self.last_action = 'none'
        self.actions = ['C', 'B', 'c', 'R', 'F']

    def is_terminal(self) -> bool:
        # If the last action is a fold then the round is over
        if self.last_action == 'none':
            return False
        if self.history[-1] == 'F':
            return True
        # Or if the street is final and the last actions were checks
        if self.last_action != 'none' and self.street == 1 and self.history[-1] in ['C', 'c']:
            return True
        return False

    def get_payoffs(self) -> int:
        if self.history[-1] == 'F':
            return self.players[(self.acting_player + 1) % 2].invested
        else:
            h_index = self.acting_player
            o_index = (self.acting_player + 1) % 2

            rank_active = GameFunctions.rank([self.cards[h_index][0],
                                              self.cards[2][0]])

            rank_opponent = GameFunctions.rank([self.cards[o_index][0],
                                                self.cards[2][0]])

            if rank_active < rank_opponent:
                return self.players[o_index].invested
            if rank_active == rank_opponent:
                return 0
            else:
                return -self.players[h_index].invested

    def get_actions(self) -> List[str]:
        if self.last_action == 'none':
            return ['c', 'B', 'F']
        if self.last_action == 'c':
            return ['c', 'B', 'F']
        if self.last_action == 'B':
            return ['R', 'C', 'F']
        if self.last_action == 'R' and self.re_raise_param < 2:
            return ['F', 'C', 'R']
        if self.last_action == 'R' and self.re_raise_param == 2:
            return ['F', 'C']

    def handle_action(self, action: str):
        new_game_state = deepcopy(self)
        new_game_state.history.append(action)
        new_game_state.last_action = action
        new_game_state.acting_player = (self.acting_player + 1) % 2

        if self.last_action != 'none':
            if action in ['C', 'c'] and self.history[-1] in ['R', 'B', 'c']:
                if self.street == 0:
                    new_game_state.street = 1
                    new_game_state.last_action = 'none'
                    new_game_state.re_raise_param = 0

            if action in ['B', 'C', 'R']:
                new_game_state.players[self.acting_player].invested += 2 + 2 * self.street

            if action == 'R':
                new_game_state.re_raise_param += 1
        else:
            if action == 'B':
                new_game_state.players[self.acting_player].invested += 2 + 2 * self.street

        return new_game_state

    def get_representation(self) -> str:
        player_card_rank = self.cards[self.acting_player][0]
        actions_as_string = "/".join([str(x) for x in self.history])
        if self.street == 0:
            community_card_rank = ''
        else:
            community_card_rank = self.cards[2][0]
        return f'{player_card_rank}/{community_card_rank}-{actions_as_string}'

class LeducCFR:
    def __init__(self):
        self.infoset_map: Dict[str, InformationSet] = {}

    def get_information_set(self, game_state: GameState) -> InformationSet:
        representation = game_state.get_representation()
        if representation not in self.infoset_map:
            self.infoset_map[representation] = InformationSet()
        return self.infoset_map[representation]

    def cfr(self, game_state: GameState, reach_probabilities: np.array) -> int:
        if game_state.is_terminal():
            return game_state.get_payoffs()

        info_set = self.get_information_set(game_state)

        strategy = info_set.get_strategy(reach_probabilities[game_state.acting_player])
        counterfactual_values = np.zeros(NUM_ACTIONS)

        for index, action in enumerate(game_state.actions):
            if action in game_state.get_actions():
                action_probability = strategy[index]

                new_reach_probabilities = reach_probabilities.copy()
                new_reach_probabilities[game_state.acting_player] *= action_probability

                counterfactual_values[index] = -self.cfr(game_state.handle_action(action), new_reach_probabilities)

        node_value = counterfactual_values.dot(strategy)

        for index, action in enumerate(game_state.actions):
            if action in game_state.get_actions():
                info_set.cumulative_regrets[index] += \
                    reach_probabilities[(game_state.acting_player + 1) % 2] * \
                    (counterfactual_values[index] - node_value)

        return node_value

    def train(self, iterations):
        util = 0
        deck = GameFunctions.get_deck_leduc()
        for _ in range(iterations):
            random.shuffle(deck)
            game_state = GameState(deck)
            reach_probabilities = np.ones(NUM_ACTIONS)
            util += self.cfr(game_state, reach_probabilities)
        return util


iterations = 100


np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

cfr_trainer = LeducCFR()
util = cfr_trainer.train(iterations)

print(f"\nRunning Kuhn Poker chance sampling CFR for {iterations} iterations")
print(f"\nExpected average game value (for player 1): {(-1./18):.3f}")
print(f"Computed average game value               : {(util / iterations):.3f}\n")

print(len(cfr_trainer.infoset_map))
print(f"History  Call Bet Check Raise Fold")
for name, info_set in sorted(cfr_trainer.infoset_map.items(), key=lambda s: len(s[0])):
    print(f"{name:3}:    {info_set.get_average_strategy()}")