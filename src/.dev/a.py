import asyncio
import time

from poke_env.player import Player, RandomPlayer
from poke_env import PlayerConfiguration, ServerConfiguration
from random import randint
from multiprocessing import Pool


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if (battle.turn) == 0:
            print('here')
        return self.choose_random_move(battle)


N = 400

async def run(i):
    server_config = ServerConfiguration(f'localhost:{8000 + i}', None)
    
    player_id = str(randint(0,1000))
    # We create two players.
    random_player = RandomPlayer(battle_format="gen8randombattle", max_concurrent_battles=1000, player_configuration=PlayerConfiguration("p1" + player_id, None), server_configuration=server_config)
    max_damage_player = MaxDamagePlayer(battle_format="gen8randombattle", max_concurrent_battles=1000, player_configuration=PlayerConfiguration("p2" + player_id, None),server_configuration=server_config)

    start = time.time()
    # Now, let's evaluate our player
    await max_damage_player.battle_against(random_player, n_battles=N)

    print(
        "Max damage player won %d / 100 battles [this took %f seconds]"
        % (max_damage_player.n_won_battles, time.time() - start)
    )

def main(i):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(run(i))

import sys
if __name__ == '__main__':
    i = int(sys.argv[1])
    main(i)
