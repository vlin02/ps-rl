from poke_env.player import Gen8EnvSinglePlayer, SimpleHeuristicsPlayer
from poke_env import PlayerConfiguration

class VsHeuristicEnv(Gen8EnvSinglePlayer):
    def __init__(self, _):

        format = "gen8randombattle"

        randId = id(10)

        print("connected - as -", randId)

        opponent = SimpleHeuristicsPlayer(
            battle_format=format,
            player_configuration=PlayerConfiguration(f"opp #{randId}", None),
        )

        super().__init__(
            opponent=opponent,
            battle_format=format,
            start_challenging=True,
            player_configuration=PlayerConfiguration(f"rl #{randId}", None),
        )
