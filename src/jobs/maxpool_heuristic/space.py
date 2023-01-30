from poke_env.environment import Gen8Pokemon, Gen8Move, Gen8Battle, MoveCategory
import numpy as np

from lib.constants import *
from lib.utils import infx

from gym.spaces import Dict, Box, Tuple, Discrete
import numpy as np

from lib.utils import infx

POKEMON_STATE_SIZE = 24 
MOVE_STATE_SIZE = 22 

move_space = Box(-np.inf, np.inf, (MOVE_STATE_SIZE,))

pokemon_space = Dict(
    state=Box(-np.inf, np.inf, (POKEMON_STATE_SIZE,)), moves=Tuple([move_space] * 4)
)

battle_space = Dict(
    team=Tuple([pokemon_space] * 6), opp_active=pokemon_space, active=Discrete(7)
)


def analyze_space(space):
    sample_poke = space["team"][0]
    infx(f'POKEMON_STATE_SIZE = {len(sample_poke["state"])}')
    sample_move = sample_poke["moves"][0]
    infx(f"MOVE_STATE_SIZE = {len(sample_move)}")


def get_n_alive(team):
    return 6 - len(team) + sum(p.fainted for p in team)


class Side:
    def __init__(
        self,
        side_conditions: list[SideCondition],
        active=Gen8Pokemon,
        p1=bool,
        team=list[Gen8Pokemon],
    ):
        self.side_conditions = side_conditions
        self.active = active
        self.p1 = p1
        self.team = team


def get_sides(battle: Gen8Battle):
    return (
        Side(
            p1=True,
            side_conditions=battle.side_conditions,
            active=battle.active_pokemon,
            team=list(battle.team.values()),
        ),
        Side(
            p1=False,
            side_conditions=battle.opponent_side_conditions,
            active=battle.opponent_active_pokemon,
            team=list(battle.opponent_team.values()),
        ),
    )


def get_hazards(side_conditions: list[SideCondition]):
    return [
        side_condition
        for side_condition in side_conditions
        if side_condition in ENTRY_HAZARDS.values()
    ]


def get_move_space(
    battle: Gen8Battle,
    ally: Side,
    opp: Side,
    poke: Gen8Pokemon,
    move: Gen8Move,
    move_idx: int,
):
    has_stab = move.type in poke.types

    category_vec = np.zeros(len(MoveCategory))
    i = move.category.value - 1
    category_vec[i] = 1

    boosts_vec = np.zeros(len(BOOST_STATS))
    if move.boosts is not None:
        for name, count in move.boosts.items():
            i = BOOST_STATS.index(name)
            boosts_vec[i] = count

    atk_multiplier = opp.active.damage_multiplier(move.type)

    targets_self = move.target == "self"

    good_hazard = (
        move.id in ENTRY_HAZARDS and ENTRY_HAZARDS[move.id] not in opp.side_conditions
    )

    good_anti_hazard = (
        move.id in ANTI_HAZARDS and len(get_hazards(ally.side_conditions)) > 0
    )

    slot_vec = np.zeros(4)
    slot_vec[move_idx] = 1

    return np.concatenate(
        [
            [
                move.base_power / 100,
                move.accuracy,
                move.expected_hits / 5,
                has_stab,
                atk_multiplier,
                targets_self,
                good_hazard,
                good_anti_hazard,
            ],
            category_vec,
            boosts_vec / 6,
            slot_vec,
        ]
    )


def get_pokemon_space(
    battle: Gen8Battle, ally: Side, opp: Side, poke: Gen8Pokemon, slot_idx: int
):

    atk_multiplier = opp.active.damage_multiplier(poke.types)
    opp_atk_multiplier = poke.damage_multiplier(opp.active.types)

    base_stats_vec = np.array([poke.base_stats[name] for name in BASE_STATS])

    boosts_vec = np.array([poke.boosts[name] for name in BOOST_STATS])

    slot_vec = np.zeros(TEAM_SIZE)
    slot_vec[slot_idx] = 1

    moves = []
    for i, move in enumerate(poke.moves.values()):
        moves.append(get_move_space(battle, ally, opp, poke, move, i))
    for _ in range(4 - len(moves)):
        moves.append(np.zeros(MOVE_STATE_SIZE))

    return dict(
        state=np.concatenate(
            [
                [
                    poke.active,
                    poke.current_hp_fraction,
                    poke.fainted,
                    atk_multiplier,
                    opp_atk_multiplier,
                ],
                base_stats_vec / 100,
                boosts_vec / 6,
                slot_vec,
            ]
        ),
        moves=moves,
    )


def get_battle_space(battle: Gen8Battle):
    p1, p2 = get_sides(battle)
    team = list(battle.team.values())

    return dict(
        team=[
            get_pokemon_space(battle, p1, p2, poke, i) for i, poke in enumerate(team)
        ],
        active=6 if p1.active is None else p1.team.index(p1.active),
        opp_active=get_pokemon_space(battle, p2, p1, p2.active, 0),
    )
