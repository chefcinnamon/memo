+++
date = '2026-10-06'
draft = false
title = 'Applied Data Science Part 2'
slug = 'ds2'
+++

## Intro {.no-counter}
Hi, this is part 2 of applied data science, following the legend Carlos Mercado.


## Regression Model {.no-counter}
Regression is a  method used to model the relationship between an outcome (or a graph of outcomes) with one or more input variables.

We choose a type of regression model and its mathematical form, then use observed data to estimate the parameters that best fit the data.



| Type | What does the model estimate? | Sports example | Example prediction |
| --- | --- | --- | --- |
| Linear | Expected numerical value | Points scored | 112.4 points |
| Logistic | Probability of an event | Will Over 210 hit? | 68% probability |
| Poisson | Expected count | Number of goals | Expected 2.3 goals |


## Linear regression
In our previous blog posts, we built a neural network from scratch following Karpathy's micrograd tutorial. Nonlinear activations such as tanh() and multiple layers allow neural networks to represent much more complicated functions. <br>
*nonlinear neural network:*
> $$
> y=\tanh(x_1w_1+x_2w_2+b)
> $$

***
<br>

In R, a function like lm() does trainging without *backpropagation* or *gradient descent*, but it tries to mimimize the error/loss, and it learns from data. <br>
It has parameters of $\beta$ coefficients	rather than weights + biases.

***

**lm()** in R tries to

$$
\boxed{\text{Find }\beta_0,\beta_1\text{ that minimize this error}}
$$

and it finds one line that is closest to the points. It uses

$$
\beta_1 =
\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}
{\sum (x_i-\bar{x})^2}
$$

and:

$$
\beta_0=\bar y-\beta_1\bar x
$$

Typically, ordinary linear regression does this by minimizing the sum of squared residuals:

$$
\sum_i (y_i-\hat y_i)^2
$$

<br>

In plain English: try to make the total squared vertical errors between the line and the observed points as small as possible.

the reason it uses squared is to make the negative errors positive, so positive and negative errors dont cancel each other out.

This highlights the difference from a nonlinear neural network: lm() is restricted to a particular form, while a neural network can represent a much more complex function.
In other words the equation in a non linear neural network is dependant on it's inputs.

***

Let's get back to main topic here **Linear Regression** with a practical example, and try to make a model.


[Full R script for the linear regression example](../../code/linear_regression.R).

<br>





## What did our volleyball model learn? {.no-counter}

Our goal: make a model that predicts a player's attack error based on their attack attempts **Here, $X$ is a player's attack attempts and $Y$ is their attack errors in the same match**.

Using **38,697 men's NCAA Division I player match records**, R learned:

$$
\boxed{\hat Y = 0.02282 + 0.17394X}
$$

| Parameter | Learned value | Standard error (SE) | Interpretation |
| --- | ---: | ---: | --- |
| Intercept, $\beta_0$ | 0.02282 | 0.01035 | Estimated errors at zero attack attempts: approximately zero. |
| Slope, $\beta_1$ | 0.17394 | 0.001820 | Ten additional attempts are associated with about **1.74 additional errors**. |

At **30 attempts**, the model estimates **5.24 errors on average**. More attempts provide more opportunities for errors; this measures error volume. To compare player efficiency, we would examine errors per attempt.

![Men's volleyball regression: attack errors versus attack attempts, with the fitted coefficients and R squared labeled.](/img/ds2-mens-regression.png)

## Where did we get the data? {.no-counter}

We downloaded NCAA statistics archived by **Jeffrey R. Stevens' [ncaavolleyballr project](https://jeffreyrstevens.github.io/ncaavolleyballr/articles/data.html)**, then organized them into our local SQLite database. The [source CSV files](https://github.com/JeffreyRStevens/ncaavolleyballr/tree/main/data-csv) include player match, team match, player season, and team season statistics.

For this example, we used men's Division I player match records dated **January 13, 2021 through May 12, 2025**. A row is **one player's performance in one match**, so the same player appears repeatedly. The men's archive filenames use years 2020 to 2024; the actual match dates determine the calendar coverage above. Invalid or ambiguous records were excluded during database preparation, and missing statistics were not filled with invented zeros.

{{< regression-playground >}}

<br>
<br>

**R² = 68.73%:** Imagine these four player match records:

| Attack attempts | Attack errors |
|---:|---:|
| 10 | 1 |
| 20 | 3 |
| 30 | 5 |
| 40 | 7 |

The average is **4 attack errors**. “Variation” describes how the counts **1, 3, 5, and 7 differ from that average**.

* **R² = 0%:** Predictions perform the same as assigning every record the average of 4.
* **R² = 100%:** The fitted line predicts every observed count exactly.
* **Our R² = 68.73%:** In the actual dataset, using attack attempts reduces the total squared prediction error by 68.73% compared with assigning every record the dataset's average.

So R² measures **how much better the fitted line describes the observed counts than the average alone**.

**RMSE (root mean squared error) = 1.279 attack errors:** measures the size of the model's misses. Square each difference between actual and estimated errors, average those squares, then take the square root: $RMSE=\sqrt{SSE/n}$. Larger misses count more heavily.

**SE (standard error):** measures uncertainty in an estimated coefficient. Here, the intercept SE is **0.01035 attack errors**, and the slope SE is **0.001820 attack errors per attempt**. These SEs account for repeated players and shared matches, using [two way clustering](https://sandwich.r-forge.r-project.org/reference/vcovCL.html). The calculation assumes errors are uncorrelated for records sharing neither a player nor a match.

For this fitted line, $SSE=\sum(y_i-\hat y_i)^2$ and $S_{xx}=\sum(x_i-\bar{x})^2$. In this example, $n=38,697$, $\bar{x}=9.444918$, $SSE=63,313.547$, and $S_{xx}=4,599,898.843$.

**SSE:** the sum of squared differences between actual attack errors and the model's estimates. It measures how much the model misses.

**Sxx:** the sum of squared differences between attack attempts and their average. It measures how spread out the attack attempt values are.

The [full R script](../../code/linear_regression.R) calculates these SEs by combining contributions grouped by player and by match, subtracting the overlap counted in both groups, and taking the square root of each coefficient variance.

Note:
`More data can increase the model accuracy`

## Logistic regression

Logistic regression models a **binary outcome**, such as winning or losing, and produces a probability between 0 and 1.

Our goal: **build a model that estimates a team's chance of winning from the difference in attack error rates between the two teams.**

There is **one input**, $X$: the team's attack error rate minus the opponent's, measured in **percentage points**. Attack error rate is $\text{attack errors}/\text{attack attempts}$. For example, a team with a 15% error rate against an opponent at 20% has $X=-5$.

Each team's error rate uses **all available earlier matches in that season**. The current match and later matches are excluded. We require at least three earlier matches with recorded attacking statistics for each team. Here, $Y=1$ means a win and $Y=0$ means a loss.

Each match is counted once. We choose which team's result to record using its ID, regardless of who won. The model assumes a **50% chance of winning when both teams have the same earlier error rate**. If one team's estimated chance is 70%, the other team's is 30%.

The [full R script](../../code/logistic_regression.R) prepares the data. The model itself is:

```r
train <- matches[matches$match_date < as.Date("2025-01-01"), ]
test <- matches[matches$match_date >= as.Date("2025-01-01"), ]

logistic_model <- glm(won ~ 0 + error_rate_gap_pp, data = train, family = binomial())
coef(logistic_model)
predict(logistic_model, test, type = "response")
```

We fix $\beta_0$ at zero so the model gives the two opposing teams win probabilities that add up to 100%.

With $\beta_0=0$, the model learns the slope, $\beta_1$, in:

$$
p=\frac{1}{1+e^{-(\beta_0+\beta_1X)}}
$$

| Parameter | Value | SE |
| --- | ---: | ---: |
| Intercept, $\beta_0$ | Fixed at 0 | Not applicable |
| Error rate difference, $\beta_1$ (per percentage point) | −0.32089 | 0.03088 |

The SE accounts for matches sharing either team, using a [dyadic robust variance estimate](https://arxiv.org/abs/1312.3398). It assumes matches with no teams in common are uncorrelated.

$$
r_i=Y_i-\hat p_i
$$

* $i$ and $j$: two different matches.
* $Y_i$: the recorded team's actual result in match $i$, with 1 for a win and 0 for a loss.
* $\hat p_i$: the model's estimated win probability for that team.
* $r_i$: the prediction error, calculated as actual result minus estimated probability.

For matches with no team in common, we assume:

$$
\operatorname{Cov}(r_i,r_j)=0
$$

* $r_j$: the prediction error for match $j$.
* $\operatorname{Cov}(r_i,r_j)$: covariance, which measures whether the two prediction errors tend to move together.
* $0$: we assume neither a positive nor a negative relationship between those errors.

The full script calculates this SE instead of R's ordinary `glm()` SE.

The negative slope means a higher attack error rate relative to the opponent is associated with a lower chance of winning. Each **1 percentage point increase** in the difference changes the **log odds** by −0.32089; the formula converts log odds into a probability.

![Men's volleyball: individual wins in green at Y equals 1 and losses in red at Y equals 0, with the fitted win probability curve.](/img/ds2-mens-logistic.png)

The horizontal axis shows the team's earlier attack error rate minus the opponent's, in **percentage points**. Small green dots show wins ($Y=1$); red dots show losses ($Y=0$). Each dot represents one of the **1,995 training matches**, at its actual coordinates. The black curve shows the model's estimated win probability.

**At 5 percentage points lower, the fitted win probability is 83%; at equal rates, 50%; at 5 points higher, 17%.** For example, earlier error rates of 15% and 20% give a difference of −5 percentage points. This is an association, not proof that reducing errors causes the stated change in winning probability.

We fit on **1,995 matches from 2021 to 2024** and test on **648 matches from 2025**. Here, $n=648$.

**SSE (sum of squared errors):** adds up the squared differences between actual results and estimated win probabilities.

$$
SSE=\sum_{i=1}^{n}(Y_i-\hat p_i)^2\approx126.0281
$$

**Brier score:** averages those squared errors for a binary outcome.

$$
\text{Brier score}=\frac{SSE}{n}\approx0.1945
$$

**RMSE (root mean squared error):** takes the square root of that average.

$$
RMSE=\sqrt{\frac{SSE}{n}}=\sqrt{\text{Brier score}}\approx0.4410
$$

Always predicting a 50% win probability gives a **Brier score of 0.2500** and **RMSE of 0.5000**. Lower is better for both; zero means perfect probability predictions.

## Poisson regression

Poisson regression estimates the average number of events, such as attack errors per match. We write this predicted average as $\lambda$.

The model learns a straight line for the [logarithm of that average](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/family.html): $\log(\lambda)=\beta_0+\beta_1x$, where $x$ is the input. To get the predicted average count, it converts back using $\lambda=e^{\beta_0+\beta_1x}$. This keeps the prediction positive.

| | Linear regression | Logistic regression | Poisson regression |
| --- | --- | --- | --- |
| Model predicts | Expected numerical value | Probability of $Y=1$ | Expected count |
| Prediction formula | $\beta_0+\beta_1x$ | $p=\frac{1}{1+e^{-(\beta_0+\beta_1x)}}$ | $\lambda=e^{\beta_0+\beta_1x}$ |
| One additional unit of $x$ | Adds $\beta_1$ to the expected outcome | Multiplies the odds of $Y=1$ by $e^{\beta_1}$ | Multiplies the expected count by $e^{\beta_1}$ |
| Prediction range | Any real number | Probability between 0 and 1 | Positive expected count |
| Typical shape with one predictor | Straight line | S shaped curve | Exponential curve |

**One example: attack errors versus attack attempts.** Both models use the same input, $X=$ attack attempts, and the same outcome, $Y=$ attack errors in the same match. This comparison uses **109,123 men's NCAA Division I and III player match records** from the [ncaavolleyballr archive](https://jeffreyrstevens.github.io/ncaavolleyballr/articles/data.html). We fit on **80,644 records from 2021 to 2024** and test on **28,479 records from 2025**. Duplicate records were removed, and conflicting records were excluded.

| Model | Fitted equation | Test RMSE |
| --- | --- | ---: |
| Linear | $\hat\mu=0.04583+0.17691X$ | 1.291 errors |
| Poisson | $\hat\lambda=e^{-0.28268+0.05739X}$ | 1.795 errors |

Both $\hat\mu$ and $\hat\lambda$ are estimated average attack error counts.

![Individual men's volleyball match counts with linear and Poisson fitted means. Small gray dots show actual 2025 attack attempts and errors, while the Poisson curve rises sharply.](/img/ds2-linear-poisson-attack-errors.png)

Small gray dots show all **28,479 actual 2025 match counts**. Both curves estimate average counts. The graph shows **0 to 131 attempts**, the full recorded range. The 2025 observations end at **87 attempts**, marked by the vertical dotted line. Only **104 test records exceed 50 attempts**. The vertical scale ends at 40 errors; labels show the larger Poisson predictions above that scale. Both models and their test RMSEs use all valid records.

Linear adds **0.177 expected errors per additional attempt**. Poisson increases its expected count by **5.9% per additional attempt**, so its curve rises increasingly sharply. At **100 attempts**, linear estimates **17.74 errors** and Poisson estimates **234.17**. At **131 attempts**, the estimates are **23.22** and **1,387.23**. Predictions beyond 87 attempts have no 2025 observations for comparison.

Attack errors cannot exceed attack attempts. The Poisson predictions at 100 and 131 attempts violate that limit, showing why this fitted equation is unsuitable at high attempt counts.

The observed averages follow an approximately straight line. **Linear has 28.1% lower test RMSE** than this Poisson model. A count outcome alone does not make the exponential relationship a better fit.

[Full R script for this comparison and graph](../../code/poisson_vs_linear.R).

## Resources {.no-counter}

[Carlos Mercado's Applied Data Science](https://everything-ds.com/)
