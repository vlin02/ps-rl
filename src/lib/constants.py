from poke_env.environment import SideCondition
from poke_env.data import GEN8_POKEDEX, GEN8_MOVES

GEN8_POKE_ID = {p: i + 1 for i, p in enumerate(GEN8_POKEDEX.keys())}
GEN8_MOVE_ID = {p: i + 1 for i, p in enumerate(GEN8_MOVES.keys())}

BOOST_STATS = ["accuracy", "atk", "def", "evasion", "spa", "spd", "spe"]
BASE_STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
POKE_STATS = ["atk", "def", "spa", "spd", "spe"]

ENTRY_HAZARDS = {
    "spikes": SideCondition.SPIKES,
    "stealhrock": SideCondition.STEALTH_ROCK,
    "stickyweb": SideCondition.STICKY_WEB,
    "toxicspikes": SideCondition.TOXIC_SPIKES,
}

ANTI_HAZARDS = {"rapidspin", "defog"}

TEAM_SIZE = 6