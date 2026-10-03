+++
date = '2026-10-02T12:22:41-04:00'
draft = true
title = 'LP, Part 1'
slug = 'lp1'
+++
## Intro {.no-counter}

Liquidity Provider (*LP*): a person, institution, or market participant that makes assets available for buying and selling, and helping a market operate smoothly. 
Liquidity providers may earn fees, spreads, or other compensation for providing this liquidity. 

LP is a fundamental building block of every financial market, including crypto (aka onchain).

Currently, I am focusing on the concept of the onchain LP, and we hope to learn, experiment and achieve some cool results.

## From orderbooks to AMMs {.no-counter}
In a traditional exchange market, there is generally a list of buy orders (bids) and a list of sell orders (asks). Together, these form the order book.
![Order book showing sell orders above the spread and buy orders below it](/img/orderbook.png)

In a traditional exchange market, liquidity providers help fill this order book by continuously placing buy and sell orders at different prices. In doing so, they make it easier for other traders to buy or sell an asset without casuing large price movements. 

For example, a liquidity provider may offer to buy an asset at USD 99.90 and sell it at USD 100.10. The difference between these prices is called the bid-ask spread, which can be one source of profit for the liquidity provider.

In traditional markets, these liquidity providers are often called **market makers**.

In 2018, Uniswap launched on Ethereum and introduced the <br> `x × y = k` <br> automated market maker (AMM) model (the original idea behind AMMs came from Vitalik on 2016 <a href="https://www.reddit.com/r/ethereum/comments/55m04x/lets_run_onchain_decentralized_exchanges_the_way/" target="_blank" rel="noopener noreferrer">[1]</a>.) <br>
And Vitalik mentioned his idea came from Nick Johnson's proposal. Worth mentioning Martin Köppelmann and Alan Lu from Gnosis suggested the `x × y = k` for the simple approach to the auto market maker. <br> <br>  

Since then, this simple formula has powered over USD 4.3 trillion in trading volume <a href="https://blog.uniswap.org/zerion-integrates-uniswap-api" target="_blank" rel="noopener noreferrer">[2]</a>.
If you are interested to read more about the Uniswap birth history, read this blog post: https://blog.uniswap.org/uniswap-history


![Uniswap constant-product curve showing how a swap changes the pool's token balances](/img/uniswap-amm-curve.png)


Unlike traditional exchanges, Uniswap did not rely on an order book filled with buy and sell orders. Instead, it introduced liquidity pools.

A liquidity pool is essentially a smart contract that holds two different assets. For example, an ETH/USDC pool might contain both ETH and USDC deposited by liquidity providers.

Instead of matching a buyer with a seller, traders swap directly against the assets inside the pool.

The price is determined automatically using the formula: <br>

`x × y = k`

where:

x is the amount of the first token in the pool,

y is the amount of the second token,

and k is a constant during trades but it recalculates when liquidity is deposited or withdrawn or fees generated.

When a trader buys one token from the pool, its supply inside the pool decreases while the supply of the other token increases. The AMM automatically adjusts the price to keep the relationship between the two reserves balanced.

For example, imagine a pool containing:

10 ETH × 20,000 USDC = 200,000

Here:

x = 10 ETH

y = 20,000 USDC

k = 200,000

The implied price of ETH inside the pool is:

20,000 USDC ÷ 10 ETH = USD 2,000 per ETH

Now imagine traders start buying ETH from the pool using USDC. As ETH leaves the pool, USDC enters it.

If the price of ETH inside the pool eventually doubles to USD 4,000, the pool could rebalance to approximately:

7.07 ETH × 28,284 USDC ≈ 200,000

The value of k remains roughly the same, and the ratio between the two assets changes.

The new implied ETH price is:

28,284 USDC ÷ 7.07 ETH ≈ USD 4,000 per ETH

Notice that the pool’s total dollar value has also changed.

Initially, the pool was worth:

10 ETH × USD 2,000 + 20,000 USDC = USD 40,000

After ETH reaches USD 4,000, the pool is worth approximately:

7.07 ETH × USD 4,000 + 28,284 USDC ≈ USD 56,568

This simple mechanism allows the pool to automatically adjust prices as traders buy and sell,

LPs deposit assets directly into a liquidity pool. In return, they receive a share of the trading fees generated whenever traders swap through that pool.

In our example the LP, deposits equal value of both tokens (if they want to create a new pool, they can set any ratio between those 2 tokens).
For instance ratio is 1eth=2000usdc

they can add 0.5eth + 1000 usdc or 10eth + 20k usdc... and so on.

-------

As you see in the x × y = k graph, the liquidity can be used by the traders in any set of token rate, from near 0 to +infinite, in both cases the LP ends up having almost only one token(it's a curve, never touches 0 in either tokens, so the pair is always tradable)! 
Imagine the LP provided 1eth+2000usdc ,

if more people buy eth, eth quantity reduces and usdc increases. 

<br>

if more people sell eth, eth quantity increases and usdc decreases.

<br>

 if everyone buys eth, there will remain no eth(close to zero) in the pool + a lot of usdc. 

 <br>

Nicely it does the maket automatically! (handling supply, demand, price movements)
<br>

--------

#### Case #1 (decreased token price): {.no-counter}

Suppose an LP deposits:

$$
5{,}000\ \mathrm{ZIP} + 20{,}000\ \mathrm{USDC}
$$

At the time of deposit, ZIP trades at:

$$
4\ \mathrm{USD/ZIP}
$$

So the ZIP side is worth:

$$
5{,}000\ \mathrm{ZIP} \times 4\ \mathrm{USD/ZIP} = 20{,}000\ \mathrm{USD}
$$

and the USDC side is worth:

$$
20{,}000\ \mathrm{USD}
$$

Therefore, the LP starts with a total position worth:

$$
20{,}000\ \mathrm{USD} + 20{,}000\ \mathrm{USD}
= \boxed{40{,}000\ \mathrm{USD}}
$$

Now suppose ZIP falls from:

$$
4\ \mathrm{USD/ZIP} \rightarrow 1\ \mathrm{USD/ZIP}
$$

As traders sell ZIP into the pool and withdraw USDC, the pool rebalances to approximately:

$$
10{,}000\ \mathrm{ZIP} + 10{,}000\ \mathrm{USDC}
$$

The new pool price is:

$$
\frac{10{,}000\ \mathrm{USDC}}{10{,}000\ \mathrm{ZIP}}
= 1\ \mathrm{USD/ZIP}
$$

The ZIP side is now worth:

$$
10{,}000\ \mathrm{ZIP} \times 1\ \mathrm{USD/ZIP}
= 10{,}000\ \mathrm{USD}
$$

and the USDC side is worth:

$$
10{,}000\ \mathrm{USD}
$$

So the total LP value becomes:

$$
10{,}000\ \mathrm{USD} + 10{,}000\ \mathrm{USD}
= \boxed{20{,}000\ \mathrm{USD}}
$$

So when ZIP falls, the LP ends up with more ZIP and less USDC.

The LP position has also lost:

$$
40{,}000\ \mathrm{USD} - 20{,}000\ \mathrm{USD}
= \boxed{20{,}000\ \mathrm{USD}}
$$

which is:

$$
\frac{20{,}000}{40{,}000} \times 100
= \boxed{50\\%}
$$

So in this example, a 75% drop in ZIP's price causes the LP position itself to fall by 50% in total dollar value.



#### Case #2 (increase token price): {.no-counter}
Start again with the same LP position:

$$
5{,}000\ \mathrm{ZIP} + 20{,}000\ \mathrm{USDC}
$$

At:

$$
4\ \mathrm{USD/ZIP}
$$

the ZIP side is worth:

$$
5{,}000\ \mathrm{ZIP} \times 4\ \mathrm{USD/ZIP}
= 20{,}000\ \mathrm{USD}
$$

and the USDC side is worth:

$$
20{,}000\ \mathrm{USD}
$$

So the starting LP value is:

$$
20{,}000\ \mathrm{USD} + 20{,}000\ \mathrm{USD}
= \boxed{40{,}000\ \mathrm{USD}}
$$

Now suppose ZIP rises from:

$$
4\ \mathrm{USD/ZIP} \rightarrow 16\ \mathrm{USD/ZIP}
$$

As traders buy ZIP from the pool using USDC, the pool rebalances to approximately:

$$
2{,}500\ \mathrm{ZIP} + 40{,}000\ \mathrm{USDC}
$$

The new pool price is:

$$
\frac{40{,}000\ \mathrm{USDC}}{2{,}500\ \mathrm{ZIP}}
= 16\ \mathrm{USD/ZIP}
$$

The ZIP side is now worth:

$$
2{,}500\ \mathrm{ZIP} \times 16\ \mathrm{USD/ZIP}
= 40{,}000\ \mathrm{USD}
$$

and the USDC side is worth:

$$
40{,}000\ \mathrm{USD}
$$

So the total LP value becomes:

$$
40{,}000\ \mathrm{USD} + 40{,}000\ \mathrm{USD}
= \boxed{80{,}000\ \mathrm{USD}}
$$

So when ZIP rises, the LP ends up with less ZIP and more USDC.

The LP position has gained:

$$
80{,}000\ \mathrm{USD} - 40{,}000\ \mathrm{USD}
= \boxed{40{,}000\ \mathrm{USD}}
$$

which is:

$$
\frac{40{,}000}{40{,}000} \times 100
= \boxed{100\\%}
$$

So in this example, a 300% increase in ZIP's price — from USD 4 to USD 16 per ZIP — causes the LP position itself to increase by 100% in total dollar value.

Hence, as an LP regardless of the LP fees you generate, if the token price goes up, you make some profit because of that, but your profit is less comparing to holding those 2 tokens (similar to case #2) that is called **"Impermanent Loss"**.

*Impermanent loss* can also occur during a token drawdown.
Compared with simply holding the same pair of tokens, an LP can end up losing more. However, compared with holding 100% of your portfolio in the declining token, providing liquidity can reduce your overall loss because part of the position is held in the paired asset, such as USDC.

For a standard AMM, the impermanent loss formula is:

$$
IL=\frac{2\sqrt{r}}{1+r}-1
$$

where:

$$
r=\frac{\text{new token price}}{\text{initial token price}}
$$

For your ZIP example:

$$
r=\frac{16}{4}=4
$$

So:

$$
IL=\frac{2\sqrt{4}}{1+4}-1
$$

$$
=\frac{4}{5}-1
$$

$$
=0.8-1
$$

$$
=\boxed{-20\\%}
$$

For the price drop from USD 4 to USD 1:

$$
r=\frac{1}{4}=0.25
$$

$$
IL=\frac{2\sqrt{0.25}}{1+0.25}-1
$$

$$
=\frac{1}{1.25}-1
$$

$$
=\boxed{-20\\%}
$$

So a 4× rise and a 4× fall produce the same 20% impermanent loss, ignoring fees.

-----
Worth noting: Uniswap V1 and V2 charged a fixed swap fee of 0.3% (30 basis points) per trade. In V2, when the protocol fee was enabled, 0.05% (5 basis points) went to the protocol.

-----
## Uniswap v3 and v4: the path to efficiency {.no-counter} 

As we’ve seen, sharp token price movements can cause capital losses for LPs in Uniswap v1 and v2, including impermanent loss.

This is where uniswap v3 was built to address the issue of capital efficiency.

Uniswap v3(introduced in 2021) allows LPs to concentrate their liquidity within a specific price range instead of providing liquidity across the entire price curve. 
<br>
For instance, an LP can set to provide 1eth+2000usdc with 1% fee in the range of -10% and +10%, if the eth price moves over 2200 or below 1800, and stops earning fees while it’s out of range<br>

When the price moves fully outside the selected range, the LP’s position is entirely denominated in one of the pool’s two tokens.

![Uniswap v3 liquidity concentrated over a selected price range along the reserve curve](/img/uniswap-v3-concentrated-range.gif)

By concentrating liquidity within a chosen price range, LPs can make more of their capital and potentially earn more fees than with full range liquidity, as long as the price stays in range and trading occurs there.

> This gave LPs more control over where they deploy their capital and enabled more custom strategies through the smart contracts.

Moreover Uniswap v3 introduced some math workarounds in their smart contracts:

sqrtPriceX96 is the square root of the raw token1/token0 price (before adjusting for token decimals), multiplied by $2^{96}$ and returned in slot0. Uniswap uses square root price in its tick and concentrated liquidity calculations.

Q64.96 means 64 integer bits and 96 fractional bits. The 64 integer bits cover the supported square root price range, roughly $2^{-64}$ to $2^{64}$. Solidity uses integer arithmetic, so the $2^{96}$ scale preserves fractional precision. Uniswap chose 96 fractional bits so sqrtPriceX96 (160 bits) and slot0's other 88 bits fit together in one 256-bit storage slot (248 bits total), reducing gas. In the tick formula, 1.0001 is the price multiplier for each tick, so each step changes the price by 0.01%.

The square root price is:

$$
\sqrt{P} = \frac{\text{sqrtPriceX96}}{2^{96}}
$$

The raw token1/token0 price ratio P is:

$$
P = \left(\frac{\text{sqrtPriceX96}}{2^{96}}\right)^2
$$

The current tick is:

$$
i_c = \left\lfloor \frac{\log(P)}{\log(1.0001)} \right\rfloor
$$

A position is in range when:

$$
i_l \leq i_c < i_u
$$

You can read more here:

<a href="https://blog.uniswap.org/uniswap-v3-math-primer" target="_blank" rel="noopener noreferrer">[Uniswap v3 math primer part 1]</a> <br>
<a href="https://blog.uniswap.org/uniswap-v3-math-primer-2" target="_blank" rel="noopener noreferrer">[Uniswap v3 math primer part 2]</a> <br>
<a href="https://atiselsts.github.io/pdfs/uniswap-v3-liquidity-math.pdf" target="_blank" rel="noopener noreferrer">[In-depth math explanation]</a>

> btw it's soo cool!

Fast forward, Uniswap v4 (draft introduced in 2023 and went live in 2025), introduced 2 major updates: <br>

1. **Hooks:** External contracts that add custom behavior at key points in a pool’s lifecycle, enabling features such as dynamic fees, creators fees, oracles, onchain limit orders and more! (Finally achieving the dream of ***money legos***, which were mainly discussed in DeFi summer 2020)

2. **Gas optimization** with various techniques such as <br>
A. a singleton **PoolManager** manages all pools, avoiding a separate contract deployment for each pool. <br>
B. **Flash accounting** nets balances across operations and settles only the final amounts, reducing intermediate token transfers—especially in multi-hop swaps.
<br>

C. **Transient storage** holds temporary accounting data for the transaction, reducing storage costs
<br>

D. **Native ETH support** avoids wrapping and unwrapping ETH where applicable
