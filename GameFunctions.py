import itertools
from typing import List


def rank(hand: List[str]) -> int:
    ranks = {
        'KK': 1,
        'QQ': 2,
        'JJ': 3,
        'KQ': 4, 'QK': 4,
        'KJ': 5, 'JK': 5,
        'QJ': 6, 'JQ': 6
    }

    cards = hand[0][0] + hand[1][0]
    return ranks[cards]


def get_deck_leduc():
    cards = ['J', 'Q', 'K']
    suits = ['h', 'c', 'd', 's']
    deck = []
    for card, suit in itertools.product(cards, suits):
        deck.append(card + suit)
    return deck
