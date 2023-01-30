from poke_env.environment import Gen8Battle
import numpy as np

BATTLE_SPACE_SIZE = 10

def get_battle_space(battle: Gen8Battle):
    base_power = np.zeros(4)
    move_type = np.zeros(4)
    for i, move in enumerate(battle.available_moves):
        base_power[i] = move.base_power / 100
        move_type[i] = move.type.value
    
    type_1 = battle.opponent_active_pokemon.type_1
    type_2 = battle.opponent_active_pokemon.type_2

    opp_type = np.array([type_1.value, 0 if type_2 is None else type_2.value])

    return np.concatenate(
        [
            base_power,
            move_type,
            opp_type
        ]
    )
