#!/usr/bin/env bash
set -e

# 1) Start two agents (in two terminals normally). If you want to background them here, uncomment:
# PORT=5008 AGENT_NAME=TronX_A python agent.py > /dev/null 2>&1 &
# PORT=5009 AGENT_NAME=TronX_B python agent.py > /dev/null 2>&1 &

# 2) Loop a bunch of games with different seeds
GAMES=${1:-50}
for ((i=1;i<=GAMES;i++)); do
  export GAME_SEED=$RANDOM
  echo "=== Match $i / $GAMES  (GAME_SEED=$GAME_SEED) ==="
  PLAYER1_URL=http://localhost:5008 \
  PLAYER2_URL=http://localhost:5009 \
  python judge_engine.py
done
