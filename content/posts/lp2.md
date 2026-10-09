+++
date = '2026-10-07'
draft = false
title = 'LP, Part 2'
slug = 'lp2'
math_inline_dollars = false
+++

## Intro {.no-counter}
As we learnt how LPing works, in this post, we learn more concepts of managing a liquidity position.

## @Delta and Gamma@ {.no-counter}
Suppose we have a position of $3000 usdc + 1 eth on Uniswap v3/v4 <br>

When ETH’s price rises within our liquidity range, our position automatically sells some ETH for USDC. When the price falls within that range, our position buys ETH with USDC.<br>

Comparing with only holding the two assets, you possibly make profit less when the price of the token surges, and you would lose more, when the price of the token fall. <br>Also know as **Impermanent loss**, option traders call that **"Cost of being short gamma"** <br>
<br>
So LPs are risking that and they hope: <u>fees > IL</u>
<br>

An in range v3/v4 uniswap position is **short gamma**, in other words, the direction of delta and price change is opposite. <br> hence the exposure changes oppositely. <br> <br>

1. when eth price goes up, LP has less eth, delta decreases, LP exposure decreases (in more human words: you miss the higher profit)
<br>

2. when eth price goes down, LP has more eth, (gamma is negative or short means the price direction is opposite of delta direction), delta increases, the LP exposure increases (in more human words: you bear higher loss)

<br>

**Delta (Δ)**: Measures how much a position's value changes, when an asset's price changes.

$$
\Delta = \frac{dV}{dP}
$$

<br>

**V** is the position's value <br>
**P** is the asset's price
<br>

For instance: **Δ = 0.50:** a small $1 rise in the asset’s price adds approximately **$0.50** to the position’s value.

<br>
<br>

**Gamma (Γ)**: Measures the magnitude and direction of delta's change per unit of asset price change.

$$
\Gamma = \frac{d\Delta}{dP} = \frac{d^2V}{dP^2}
$$

Here, **Δ** is delta, **P** is the asset's price, and **V** is the position's value.

For instance, if the asset's price rises by **$10** and delta falls from **0.50 to 0.40**:

$$
\text{Average gamma} = \frac{0.40-0.50}{10} = -0.01
$$

### Gamma signs {.no-counter}

This first playground uses an illustrative delta equation to compare positive, negative, and zero gamma.

{{< lp-playground mode="signs" >}}

### Delta and gamma in an LP {.no-counter}

This playground calculates our ETH/USDC position using Uniswap's liquidity equations. Moving the price shows how the position's value, token balances, delta, and gamma change.

{{< lp-playground mode="position" >}}

--------

Since LPing is very similar to option trading, we need to get familiar with options, firsly let's compare options with perpetual futures (perps)

| Feature | Perpetual futures | Options |
| --- | --- | --- |
| Expiration | None | Usually yes |
| Upfront premium | No | Buyer pays premium |
| Price exposure | Linear | Nonlinear |
| Maximum loss | Can be very large | Limited to premium for buyers |
| Funding | Usually periodic | Depends on contract type |
| Liquidation | Yes, with leverage | Possible for sellers |


<br>
For instance:
<br>

**Long perp:** Open long → ETH rises to $120 → you gain $20

ETH falls to $80 → you lose $20.


**Buy call:** Strike $100, premium $5. ETH rises to $120 → you profit $15. ETH falls to $80 → you lose only $5.

![Profit and loss comparison for a long perpetual future and a bought call option as ETH's price changes.](/img/option-perps.svg) <br>

## @Options@ {.no-counter}
in options trading there are 2 positions to enter: **Call** and **Put**

**Call**: You make profit when the price goes up. <br>
**Put**: You make profit when the price goes down. <br>
**Premium**: The cost of option contract. <br>
**Strike price**: The price at which the option lets you buy or sell.
**Expiration**: The date the option contract ends. <br>
**Notional**: The total underlying asset.

Unlike perps, you just buy the rights by paying a premium. <br>
Assume ETH is $100, you buy a right of eth of 130, call, notional: 1eth by paying a $5 premimum.<br>

Now if the eth price goes to $150, you get the right to buy eth at $130, max 1eth, and enjoy the discount! or sell your right for the profit!
<br>

In option trading, either you are buying a call/put, or you are selling a call/put, when you are a seller, you risk larger loss for that premium that buyers pay you! (Very close concept as LP! as an LP you are a seller of put )

-------
Now if you look at a uniswap v3/v4 position range, you will find it is very similar to this option strategy!: **Fig Leaf**, that you sell put at position B.

![Fig Leaf option strategy: profit and loss diagram and the setup using a bought LEAPS call at strike A and a sold short term call at strike B.](/img/big-leaf-option.png)


-------


## @v3/v4 pools@ {.no-counter}

Both charts show the same LP position. Moving the price shows how its profit and token balances change.

{{< lp-playground mode="payoff" >}}


Let's try some LPing strategy
## @Warming up {.no-counter}

As we learnt previously, **Delta** How much your position gains or loses when ETH moves $1, and **Gamma** shows How much that slope changes when ETH moves.
<br>
Let's calculate our Delta of this position: <br>

> 0.5 ETH at 100 USDC + 50 USDC, with a ±25% multiplicative range (80 to 125).

The opening price is 100 USDC. [Uniswap's price ticks](https://blog.uniswap.org/uniswap-v3-math-primer#what-does-a-tick-represent) are spaced multiplicatively, rather than by equal price amounts. Here, **±25% multiplicative** means dividing and multiplying by the same factor, **1 + 0.25 = 1.25**:

<div>
$$
\begin{aligned}
\text{Lower boundary} &= 100 \div 1.25 = 80\\
\text{Upper boundary} &= 100 \times 1.25 = 125
\end{aligned}
$$
</div>

This gives **80 to 125**, or **−20% to +25%** in ordinary percentages, matching the charts above.

Inside the range, ignoring fees, delta equals the ETH held in the position:

$$
\Delta(P) = \frac{dV}{dP}
= L\left(\frac{1}{\sqrt{P}}-\frac{1}{\sqrt{P_b}}\right)
$$

**V**: position value in USDC.<br>
**P**: current ETH price in USDC.<br>
<strong>P<sub>b</sub></strong>: upper range boundary, 125.<br>
**L**: liquidity, a fixed number that determines the position's size.

First, calculate **L** from the opening 0.5 ETH:

$$
L = \frac{0.5}{1/\sqrt{100}-1/\sqrt{125}}
\approx 47.3607
$$

The USDC required at opening is:

$$
\text{USDC} = L\left(\sqrt{100}-\sqrt{80}\right)
\approx 50
$$

So this LP opens with **0.5 ETH + 50 USDC**, worth **100 USDC**. These token amounts follow [Uniswap's liquidity equations](https://blog.uniswap.org/uniswap-v3-math-primer-2).

Substitute the opening price into the delta equation:

<div>
$$
\begin{aligned}
\Delta(100)
&= L\left(\frac{1}{\sqrt{100}}-\frac{1}{\sqrt{125}}\right)\\
&\approx 47.3607(0.1-0.0894427)\\
&\approx 0.5
\end{aligned}
$$
</div>

At opening, **delta is 0.5 ETH**: a small 1 USDC rise in ETH's price adds approximately 0.5 USDC to the LP's value. The derivative is shown in the [delta derivation](https://ledgerjournal.org/ojs/ledger/article/download/389/281/2162#page=14).



## <span class="lp-strategy-ascii" aria-hidden="true">@#&#42;%=!?8@</span> <span class="lp-strategy-label">Some LP strategy</span> <span class="lp-strategy-ascii lp-strategy-ascii-right" aria-hidden="true">@/%&amp;?!&#42;</span> {#some-lp-strategy .no-counter .lp-strategy-title}
### #1 Hedge against small price movements {.no-counter .lp-strategy-heading}

We can hedge the existing LP by borrowing 0.5 ETH and selling it for 50 USDC at price 100. When ETH's price falls, repaying the ETH debt costs less USDC, offsetting part of the LP's loss.

{{< lp-playground mode="hedge" >}}

As you see, we by hedging with delta (negative initial delta from the short position), we could reduce our starting exposure, which can be really useful when we want to focus on earing fees and when the token is trading at a narrow price range, otherwise, in bigger price movements we are losing at both sides.
> note that **gamma** stayed the same when comparing to a regular LP position.


### #2 Hedge with buying a put, strike LESS than lp begins {.no-counter .lp-strategy-heading}

An LP has price exposure similar to selling a put. A **long put (buy put)** at a lower strike can limit the LP's losses from a large price fall. Both legs use the same ETH notional. Fees and option costs are excluded.

{{< lp-playground mode="put-spread" strike="80" >}}

{{< lp-playground mode="put-spread" strike="125" >}}

{{< lp-playground mode="put-spread" strike="100" >}}


### #3 Buy a put at the same strike with a different range {.no-counter .lp-strategy-heading}

Both strikes stay at 100, and the LP range stays at 80 to 125. Only the buy put's range changes. Here, “calendar spread” compares range widths; these perpetual puts do not expire. Fees and option costs are excluded.

{{< lp-playground mode="put-range" width="100" >}}

{{< lp-playground mode="put-range" width="5" >}}

Hence, with a narow range buying put, at the same strike as LP, we can hedge nicely on both sides.

### #4 Add a short call at the LP strike {.no-counter .lp-strategy-heading}

For this comparison, the LP is modeled as a short put (sell put). Adding a short call (sell call) at the same strike forms a short straddle. Both strikes are 100 and both ranges are 80 to 125.

{{< lp-playground mode="short-straddle" >}}

## Resources {.no-counter}

[Panoptic: How to Hedge a Uniswap LP Position and Earn More](https://panoptic.xyz/docs/product/uniswap-lps/hedge)

[The Options Playbook: Fig Leaf](https://www.optionsplaybook.com/option-strategies/leveraged-covered-call)
