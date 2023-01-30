ps:
    kill -9 $(lsof -ti:8000) || true
    cd pokemon-showdown; node pokemon-showdown start --no-security & 