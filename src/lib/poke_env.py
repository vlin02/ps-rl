from poke_env.player import Player, ForfeitBattleOrder, RandomPlayer
class DummyPlayer(Player):
    def __init__(self, ret, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ret = ret
         
    def choose_move(self, battle):
        self.ret['battle'] = battle
        return ForfeitBattleOrder()

async def sample_battle():
    ret = {}

    random_player = RandomPlayer(battle_format="gen8randombattle")
    dummy_player = DummyPlayer(ret, battle_format="gen8randombattle")
    await dummy_player.battle_against(random_player, n_battles=1)

    return ret['battle']