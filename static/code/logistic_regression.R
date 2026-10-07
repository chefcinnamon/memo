library(DBI)
library(RSQLite)
folder <- path.expand("~/Desktop/volleyball-projections")
con <- dbConnect(SQLite(), file.path(folder, "data/volleyball.sqlite"), flags = SQLITE_RO)
raw <- dbGetQuery(con, "
  SELECT match_key, team_key, opponent_key, match_date, won,
         attack_errors, attack_attempts
  FROM team_match_stats WHERE sport = 'mvb' AND division = 1
")
dbDisconnect(con)
raw$match_date <- as.Date(raw$match_date)
raw$season <- format(raw$match_date, "%Y")
by_team <- split(raw, paste(raw$season, raw$team_key, sep = "|"))

# Compute attack-error rate from all known earlier matches in the season.
history <- function(team, season, day) {
  past <- by_team[[paste(season, team, sep = "|")]]
  if (is.null(past)) return(NULL)
  past <- past[past$match_date < day, ]
  past <- past[complete.cases(past[c("attack_errors", "attack_attempts")]), ]
  past <- past[past$attack_attempts > 0, ]
  if (nrow(past) < 3) return(NULL)
  list(attack_error_rate = sum(past$attack_errors) / sum(past$attack_attempts),
       n = nrow(past), latest = max(past$match_date))
}
rows <- lapply(seq_len(nrow(raw)), function(i) {
  game <- raw[i, ]
  team <- history(game$team_key, game$season, game$match_date)
  opponent <- history(game$opponent_key, game$season, game$match_date)
  if (is.null(team) || is.null(opponent)) return(NULL)
  data.frame(match_key = game$match_key, team_key = game$team_key,
             opponent_key = game$opponent_key,
             match_date = game$match_date, won = game$won,
             # Difference is measured in percentage points.
             error_rate_gap_pp = 100 * (team$attack_error_rate - opponent$attack_error_rate),
             team_prior_matches = team$n,
             team_history_latest_date = team$latest,
             opponent_history_latest_date = opponent$latest)
})
matches <- na.omit(do.call(rbind, rows))
stopifnot(!anyDuplicated(matches[c("match_key", "team_key")]),
          all(matches$team_history_latest_date < matches$match_date),
          all(matches$opponent_history_latest_date < matches$match_date))
# Keep one team perspective per match, chosen by ID rather than by the result.
matches <- matches[order(matches$match_key, matches$team_key), ]
matches <- matches[!duplicated(matches$match_key), ]
train <- matches[matches$match_date < as.Date("2025-01-01"), ]
test <- matches[matches$match_date >= as.Date("2025-01-01"), ]
# A zero intercept ensures that swapping teams gives complementary probabilities.
logistic_model <- glm(won ~ 0 + error_rate_gap_pp, data = train, family = binomial())
# Dyadic robust SE: allow correlation between matches sharing either team.
# Source: https://arxiv.org/abs/1312.3398, Appendix E.
score <- train$error_rate_gap_pp * (train$won - fitted(logistic_model))
team_scores <- tapply(c(score, score), c(train$team_key, train$opponent_key), sum)
pair <- paste(pmin(train$team_key, train$opponent_key),
              pmax(train$team_key, train$opponent_key), sep = "|")
pair_scores <- tapply(score, pair, sum)
information <- sum(train$error_rate_gap_pp^2 * fitted(logistic_model) * (1 - fitted(logistic_model)))
score_variance <- sum(team_scores^2) - sum(pair_scores^2)
stopifnot(score_variance > 0, information > 0,
          !anyDuplicated(matches$match_key),
          !length(intersect(train$match_key, test$match_key)))
robust_se <- sqrt(score_variance) / information
coefficients <- data.frame(parameter = "Attack error rate difference (per percentage point)",
                           estimate = unname(coef(logistic_model)[1]), SE = robust_se)
print(coefficients, row.names = FALSE)
cat("Intercept fixed at zero | Eligible matches:", nrow(matches),
    "| Training:", nrow(train), "| Test:", nrow(test), "\n")
p <- predict(logistic_model, test, type = "response")
results <- data.frame(
  model = c("50% probability", "Attack-error-rate difference"),
  Brier = c(mean((test$won - 0.5)^2), mean((test$won - p)^2))
)
print(results, row.names = FALSE)
saveRDS(matches, file.path(folder, "data/mvb_logistic_features.rds"))
saveRDS(list(model = logistic_model, coefficients = coefficients, results = results),
        file.path(folder, "data/mvb_logistic_results.rds"))

# Show individual binary results at their true coordinates, including overlaps.
stopifnot(all(train$won %in% c(0, 1)))
span <- range(train$error_rate_gap_pp) + c(-0.5, 0.5)
x <- seq(span[1], span[2], length.out = 400)
landmarks <- c(-5, 0, 5)
landmark_p <- predict(logistic_model, data.frame(error_rate_gap_pp = landmarks), type = "response")
draw_graph <- function() {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  par(mar = c(5.6, 5.1, 4.1, 1.2), las = 1, bty = "l", cex = 1.08)
  plot(x, predict(logistic_model, data.frame(error_rate_gap_pp = x), type = "response"),
       type = "n", xlim = span, ylim = c(-0.045, 1.045), xaxs = "i", yaxs = "i",
       xaxt = "n", yaxt = "n", xlab = "", ylab = "Match result and win probability",
       main = "Earlier attack error rates and winning")
  axis(2, at = seq(0, 1, by = 0.25), labels = c("0", "0.25", "0.50", "0.75", "1"))
  ticks <- pretty(span, n = 6)
  ticks <- ticks[ticks >= span[1] & ticks <= span[2]]
  axis(1, at = ticks, labels = ifelse(ticks > 0, paste0("+", ticks), as.character(ticks)))
  abline(h = seq(0, 1, by = 0.25), col = "#EEEEEE")
  abline(v = 0, col = "gray80", lty = 3)
  colors <- ifelse(train$won == 1, "forestgreen", "firebrick")
  points(train$error_rate_gap_pp, train$won, pch = 16, cex = 0.5,
         col = adjustcolor(colors, alpha.f = 0.2))
  lines(x, predict(logistic_model, data.frame(error_rate_gap_pp = x), type = "response"), lwd = 3)
  segments(landmarks, 0, landmarks, landmark_p, col = "gray65", lty = 3)
  text(landmarks + c(0, 0.9, 0), landmark_p + c(0.065, 0.055, -0.065),
       labels = paste0(round(100 * landmark_p), "%"), font = 2, cex = 1.1)
  mtext(sprintf("Training matches = %s", format(nrow(train), big.mark = ",")),
        side = 3, line = 0.4, cex = 0.8)
  legend("topright", inset = c(0.025, 0.14),
         legend = c("Fitted win probability", "Win (Y = 1)", "Loss (Y = 0)"),
         lty = c(1, NA, NA), lwd = c(3, NA, NA), pch = c(NA, 16, 16),
         col = c("black", "forestgreen", "firebrick"), bty = "n", cex = 0.9,
         pt.cex = c(1, 0.6, 0.6))
  mtext("Team's attack error rate minus opponent's (percentage points)", side = 1, line = 2.5)
  mtext("Lower than opponent", side = 1, line = 4, at = mean(c(span[1], 0)), col = "#666666", cex = 0.8)
  mtext("Higher than opponent", side = 1, line = 4, at = mean(c(0, span[2])), col = "#666666", cex = 0.8)
}
graph <- file.path(folder, "graphs/mvb_logistic_regression.png")
dir.create(dirname(graph), recursive = TRUE, showWarnings = FALSE)
png(graph, width = 1600, height = 1050, res = 150)
tryCatch(draw_graph(), finally = dev.off())
if (interactive()) draw_graph()
cat("Graph saved:", graph, "\n")
