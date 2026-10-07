library(DBI)
library(RSQLite)
sport <- "mvb"
folder <- path.expand("~/Desktop/volleyball-projections")

con <- dbConnect(SQLite(), file.path(folder, "data/volleyball.sqlite"), flags = SQLITE_RO)
games <- tryCatch(dbGetQuery(con, "
  SELECT player_key, match_key, attack_attempts, attack_errors FROM player_match_stats
  WHERE sport = ?
", params = list(sport)), finally = dbDisconnect(con))
games <- games[complete.cases(games[c("attack_attempts", "attack_errors")]), ]
# Missing statistics are not recorded zeros; clustering requires complete identifiers.
stopifnot(!anyNA(games[c("player_key", "match_key")]),
          all(nzchar(games$player_key)), all(nzchar(games$match_key)))

model <- lm(attack_errors ~ attack_attempts, data = games)
# Two-way clustered SEs allow dependence within players and within matches.
# HC1 correction and cluster adjustments follow:
# https://sandwich.r-forge.r-project.org/reference/vcovCL.html
X <- model.matrix(model)
scores <- X * residuals(model)
cluster_component <- function(group) {
  G <- length(unique(group))
  stopifnot(G > 1)
  crossprod(rowsum(scores, group, reorder = FALSE)) * G / (G - 1)
}
overlap <- interaction(games$player_key, games$match_key, drop = TRUE)
meat <- cluster_component(games$player_key) + cluster_component(games$match_key) -
        cluster_component(overlap)
bread <- solve(crossprod(X))
cluster_vcov <- (nrow(games) - 1) / (nrow(games) - ncol(X)) * bread %*% meat %*% bread
stopifnot(all(diag(cluster_vcov) > 0))
coefficients <- data.frame(estimate = coef(model), SE = sqrt(diag(cluster_vcov)))
print(coefficients)
cat("Records:", nrow(games), "| R-squared:", signif(summary(model)$r.squared, 3), "\n")
cat("Overall error rate:", round(100 * sum(games$attack_errors) / sum(games$attack_attempts), 2),
    "% | Expected errors at 30 attempts:", round(predict(model, data.frame(attack_attempts = 30)), 2), "\n")

# Aggregate overlapping observations; larger circles mean more match records.
dots <- aggregate(list(n = rep(1, nrow(games))), games[c("attack_attempts", "attack_errors")], sum)
draw_graph <- function() {
  plot(dots$attack_attempts, dots$attack_errors, pch = 21,
       cex = 0.4 + 2 * sqrt(dots$n / max(dots$n)),
       col = "steelblue", bg = adjustcolor("steelblue", alpha.f = 0.3),
       xlab = "Attack attempts (X)", ylab = "Attack errors (Y)",
       main = paste(if (sport == "mvb") "Men's" else "Women's",
                    "volleyball: attack attempts vs errors"))
  abline(model, col = "tomato", lwd = 3)
  legend("topleft", c(sprintf("Y = %.5f %+.5f X", coef(model)[1], coef(model)[2]),
                      sprintf("Intercept (beta0) = %.5f", coef(model)[1]),
                      sprintf("Slope (beta1) = %.5f", coef(model)[2]),
                      sprintf("R-squared = %.3f | n = %s", summary(model)$r.squared,
                              format(nrow(games), big.mark = ",")), "Circle size = record count"),
         col = c("tomato", rep("gray30", 3), "steelblue"),
         lty = c(1, NA, NA, NA, NA), pch = c(NA, NA, NA, NA, 21),
         bty = "o", bg = "white", box.col = "gray85", cex = 0.85)
}
graph <- file.path(folder, "graphs", paste0(sport, "_attack_errors_regression.png"))
dir.create(dirname(graph), recursive = TRUE, showWarnings = FALSE)
png(graph, width = 1400, height = 1000, res = 140)
tryCatch(draw_graph(), finally = dev.off())
if (interactive()) draw_graph()
cat("Graph:", graph, "\n")
