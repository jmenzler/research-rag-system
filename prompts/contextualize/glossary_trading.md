# Quant finance / market microstructure glossary

## Models & strategies
- A-S, AS = Avellaneda-Stoikov inventory-aware market making (2008)
- GLFT = Guéant-Lehalle-Fernandez-Tapia optimal liquidation
- HJB = Hamilton-Jacobi-Bellman equation (stochastic control)
- HFT = High-Frequency Trading
- MM = Market Maker / Market Making
- MFG = Mean-Field Game
- LO = Limit Order; MO = Market Order
- POV = Percentage-of-Volume execution
- TWAP / VWAP = Time / Volume-Weighted Average Price
- MLE = Maximum Likelihood Estimator

## Microstructure signals & metrics
- NBBO = National Best Bid and Offer
- BBO = Best Bid/Offer
- OB / LOB = Order Book / Limit Order Book
- DOM = Depth of Market
- OBI = Order Book Imbalance (volume-based: bid_vol vs ask_vol)
- OFI = Order Flow Imbalance (signed aggressive volume)
- VPIN = Volume-Synchronized Probability of Informed Trading
- microprice = volume-weighted mid (better short-term predictor than mid)
- reservation price = MM's inventory-adjusted internal fair value
- effective spread vs realized spread vs adverse selection (decomposition)
- queue position = priority within an LO at a given price
- maker = adds liquidity (resting LO); taker = removes liquidity (MO)
- adverse selection = MM picked off by informed traders
- toxic flow = informed flow against MM quotes
- Kyle's lambda = price impact per unit signed volume

## DeFi / AMM
- AMM = Automated Market Maker
- CLMM = Concentrated Liquidity Market Maker (Uniswap v3, Orca, Raydium CLMM)
- DLMM = Dynamic Liquidity Market Maker (Meteora bin-based)
- LP = Liquidity Provider
- LVR = Loss-Versus-Rebalancing (Milionis-Moallemi-Roughgarden)
- IL = Impermanent Loss
- MEV = Maximal Extractable Value (frontrunning, sandwiching, arbitrage)
- CEX / DEX = Centralized / Decentralized Exchange
- TVL = Total Value Locked
- bonding curve / AMM curve = e.g. x*y=k (constant product) or variants

## Risk / volatility
- EWMA = Exponentially Weighted Moving Average
- GARCH = Generalized Autoregressive Conditional Heteroskedasticity
- HAR-RV = Heterogeneous Autoregressive Realized Volatility
- RV = Realized Volatility

(Note: bare Greek symbols like γ, κ, λ, α, σ are intentionally NOT defined here — their meaning is paper-specific. Always read the meaning from the chunk's surrounding text, never assume.)

## Sizing / position
- inventory q = current position quantity
- q_normalized = (position - target) / max_position, ranges [-1, +1]
- target position / max position = MM operational limits
- ternary search = numerical optimum over a single-peaked objective

## Backtesting / optimization
- WFO = Walk-Forward Optimization
- IS / OOS = In-Sample / Out-of-Sample
- Sharpe / Sortino = risk-adjusted return ratios
- DD / MDD = Drawdown / Maximum Drawdown

## ML / RL
- RL = Reinforcement Learning
- SAC = Soft Actor-Critic (continuous-action RL)
- PPO = Proximal Policy Optimization
- DQN = Deep Q-Network

## Execution / units
- RTT = Round-Trip Time (latency)
- bps = basis points (1 bps = 0.01% = 0.0001)
- PnL = Profit and Loss
