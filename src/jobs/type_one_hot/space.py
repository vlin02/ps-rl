from matplotlib.style import available
from poke_env.environment import Gen8Battle, PokemonType, Gen8Move
import numpy as np

TYPE_SPACE_SIZE = 18
MOVE_SPACE_SIZE = 19
BATTLE_SPACE_SIZE = 112

def get_type_space(type: PokemonType):
    type_space = np.zeros(len(PokemonType))
    type_space[type.value - 1] = 1
    return type_space


def get_move_space(move: Gen8Move):
    return np.concatenate([[move.base_power / 100], get_type_space(move.type)])


def get_battle_space(battle: Gen8Battle):
    
    moves = []
    for move in battle.available_moves:
        moves.append(get_move_space(move))
    moves.append(np.zeros(MOVE_SPACE_SIZE * (4 - len(moves))))
    moves = np.concatenate(moves)

    opp_type = []
    for type in battle.opponent_active_pokemon.types:
        if type is not None:
            opp_type.append(get_type_space(type))
    opp_type.append(np.zeros(TYPE_SPACE_SIZE * (2 - len(opp_type))))
    opp_type = np.concatenate(opp_type)

    return np.concatenate([moves, opp_type])


def analyze(battle: Gen8Battle):
    global TYPE_SPACE_SIZE
    global MOVE_SPACE_SIZE
    global BATTLE_SPACE_SIZE

    sample_move = Gen8Move("struggle")
    sample_type = PokemonType.BUG

    TYPE_SPACE_SIZE = len(get_type_space(sample_type))
    MOVE_SPACE_SIZE = len(get_move_space(sample_move))
    BATTLE_SPACE_SIZE = len(get_battle_space(battle))

    print("TYPE_SPACE_SIZE", "=", TYPE_SPACE_SIZE)
    print("MOVE_SPACE_SIZE", "=", MOVE_SPACE_SIZE)
    print("BATTLE_SPACE_SIZE", "=", BATTLE_SPACE_SIZE)
