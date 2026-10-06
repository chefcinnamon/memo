+++
date = '2026-10-05'
draft = false
title = 'Applied Data Science Part 1'
slug = 'ds1'
+++

## Intro {.no-counter}
Hi, this is part 1 of applied data science, following the legend Carlos Mercado.

Probabilty, profit maxxing, game theory, math modeling, optimum result, all are related to applied data science, so here we go... 


## Excpected Value (EV)
EV is an average probability you expect to win or loss in 1 or many bets/trades/...

Suppose in a fair coin flip, since the prob. of heads and tails is 50% and same, the EV is 0, we call this **"Breakeven"** 
<br>
> breakeven = EV of zero
<br>
<br>

`Payout: total money return`
<br>
For instance: in a coin flip, the payout of <span>$1</span> deposit is <span>$2</span> and EV is 0

<br>

EV can be calculated with this equation:
<br>

$$
\mathrm{EV} = \text{Expected money returned} - \text{money you paid}
$$

<br>

$$
\mathrm{EV} = P(\mathrm{win}) \times \mathrm{Profit} - P(\mathrm{loss}) \times \mathrm{Stake}
$$

<br>

*when you spend <span>$1</span>*: <br>

$$
\mathrm{EV} = P(\mathrm{win}) \times \mathrm{Profit} - 1
$$

<br>

*generally:*: <br>

$$
\mathrm{EV} = P(\mathrm{win}) \times \text{Total Payout} - \mathrm{Stake}
$$


<br>


When the EV is positive, you expect to make profit, 

<br> and when EV is negative, you expect to loss.
<br><br>

**Odds**: Odds show the multiplier in your payout. For instance odds of 5 means payout: 5x, and **implied prob.** of that is $\frac{1}{\mathrm{odds}}$, so $\frac{1}{5} = 20\\%$, this means you need to win 20% of the time to **break even**.
<br>

-----

Example #2: Suppose a coin flip bet, although the coin heads/tails= 50% (true prob.), the house says the payout odds is x1.90 instead of 2 because of the house/platform fees, so the implied prob is $\frac{1}{1.9} \approx 0.5263$ , this means you need to win over 52.63% of the time to break even.
so the house edge is 2.63% (True Prob. − Implied Prob.), <br>
hence house is winning 2.63% on all trades theoretically. 

and your EV in case you bet <span>$1</span> in example 2 is:

$$
.50 \times 1.9 - 1 = -\\$0.05
$$

same as <br>

$$
(0.5 \times 0.9) - (0.5 \times 1) = -\\$0.05
$$

<br>
Result: House/platform tries to make EV more negative (user loss), but not too much to make the users leave the house!!
<br>
<br>

Let's model this in R:

`If I run a betting platform with a $100,000 starting bankroll, offer 2×, 1.9×, and 1.8× payouts on 50/50 bets, and users randomly bet between $1 and $500, what will my bankroll be after 1000 bets?`
<br>

```R
start <- 100000
n <- 1000
payouts <- c(2.00, 1.90, 1.80)

# House's gain or loss on each bet
house_change <- function(payout) {
  bet <- sample(500, n, replace = TRUE)
  player_wins <- runif(n) < 0.5
  ifelse(player_wins, -bet * (payout - 1), bet)
}

# Bankroll paths: one column per payout, starting at `start`
bankroll <- start + sapply(payouts, function(p) c(0, cumsum(house_change(p))))

matplot(bankroll, type = "l", lty = 1, col = c("gray40", "blue", "forestgreen"),
        xlab = "Number of bets", ylab = "House bankroll ($)")
abline(h = start, col = "red", lty = 2)
legend("topleft", paste0(payouts, "x payout"),
       col = c("gray40", "blue", "forestgreen"), lty = 1, bty = "n")

round(bankroll[n + 1, ], 2)
```
![Bankrolls over 1,000 bets for three independent betting platforms with 2.00x, 1.90x, and 1.80x total payouts; the red dashed line marks the starting bankroll of $100,000.](/img/ds1-bankroll.png)



## Overround (Total Vig or Vigorish) {.no-counter}

Overround is the sum of all implied prob. minus 100%.

$$
\boxed{\text{Overround} = \sum \text{Implied prob.} - 100\\%}
$$

When there is more than 1 outcome in an event (for instance, flipping a coin: up odds 1.9×, impl. prob. 52.63%; down odds 1.9×, impl. prob. 52.63%), we can calculate the total overround of the system, also loosely known as **total house edge**, or the **diff between the total impl. prob. and total true prob. (100%)**.

$$
\left(\frac{1}{1.9} + \frac{1}{1.9} - 1\right) \times 100\\%
\approx 5.26\\%
$$

We can calculate **EV for one outcome**; for **overround**, we calculate for **all outcomes of an event**.

EV and overround are not the same, but they are closely related.

## Resources {.no-counter}

[Carlos Mercado's Applied Data Science](https://everything-ds.com/)
