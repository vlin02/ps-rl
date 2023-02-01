from poke_env.player import Gen8EnvSinglePlayer, SimpleHeuristicsPlayer
from poke_env import PlayerConfiguration, ServerConfiguration

from lib.utils import infx

class VsHeuristicEnv(Gen8EnvSinglePlayer):
    def __init__(self, cfg):

        format = "gen8randombattle"
        randId = id(10) +cfg.vector_index
        infx("connected - as -", randId)
        infx(cfg.vector_index)

        # server_configuration = ServerConfiguration(f"localhost:{8000 + cfg.vector_index}", None)
        server_configuration = None

        opponent = SimpleHeuristicsPlayer(
            battle_format=format,
            player_configuration=PlayerConfiguration(f"opp{randId}", None),
            server_configuration=server_configuration,
            max_concurrent_battles=200
        )

        super().__init__(
            opponent=opponent,
            battle_format=format,
            start_challenging=True,
            player_configuration=PlayerConfiguration(f"rl{randId}", None),
            server_configuration=server_configuration,
        )
