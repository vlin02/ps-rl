from jobs.maxpool_heuristic.space import get_battle_space, analyze_space
from lib.poke_env import sample_battle
import asyncio

async def main():
    for i in range(1):
        battle = await sample_battle()
        analyze_space(get_battle_space(battle))

if __name__ == "__main__":
    asyncio.new_event_loop().run_until_complete(main())