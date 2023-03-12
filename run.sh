for ((i=10; i<30; i++))
do
    cd /home/ubuntu/ps-rl/ps$i;
    kill -9 $(lsof -ti:$(expr 8000 + $i)) || true
    node pokemon-showdown start --no-security --port $(expr 8000 + $i)  &
done