ps:
    kill -9 $(lsof -ti:8000) || true
    cd pokemon-showdown; node pokemon-showdown start --no-security & 
tb:
    tensorboard --logdir=out/result    
ray:
    code /Users/vlin/miniconda3/envs/ps-rl/lib/python3.10/site-packages/ray

ps1:
    kill -9 $(lsof -ti:8001) || true
    cd pokemon-showdown; node pokemon-showdown start --port 8001  --no-security & 

ps2:
    kill -9 $(lsof -ti:8002) || true
    cd pokemon-showdown; node pokemon-showdown start --port 8002  --no-security & 