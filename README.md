# DRLTrade
try to train the network for trade, using PPO, RL, LSTM

- internal data source
- base on 1min ohlcv data

## PPO2
- assum that we enter the market at any time and exit after 500 bar
- we decide how much position we should hold after every minutes, 3 action is allowed. hold 1 long pos, hold 1 short pos or hold 0 pos
- LSTM helps to figure out the trading continuity during this 500 bar
- one hand we use actor to make the decision , the other hand we use critic to adjust the params
- final goal is to maximize the pnl after 500 bar
- profit in the future should be considered less attractive, just like the cash flow discount
- commision need to be set, but no slippage
