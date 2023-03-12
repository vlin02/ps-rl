for ((i=10; i<30; i++))
do
    cd /home/ubuntu/ps-rl/src;
    python3 .dev/a.py $i &
done