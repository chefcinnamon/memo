# One real men's volleyball example: attack errors versus attack attempts.
# Source: https://jeffreyrstevens.github.io/ncaavolleyballr/articles/data.html
library(DBI)
library(RSQLite)
folder <- path.expand("~/Desktop/volleyball-projections")
database <- file.path(folder, "data/mens_regression_comparison.sqlite")
if (!file.exists(database)) stop("The men's regression database is missing: ", database)
con <- dbConnect(SQLite(), database, flags = SQLITE_RO)
games <- tryCatch(dbGetQuery(con, "
  SELECT division, player_key, team_key, match_key, match_date,
         attack_attempts, attack_errors
  FROM player_match_stats WHERE division IN (1, 3)
"), finally = dbDisconnect(con))
games$match_date <- as.Date(games$match_date)
# Keep genuine counts, including zeros. Missing statistics are excluded.
n_input <- nrow(games)
games <- games[complete.cases(games) & games$attack_attempts >= 0 &
               games$attack_errors >= 0 & games$attack_errors <= games$attack_attempts, ]
games <- games[games$attack_attempts == floor(games$attack_attempts) &
               games$attack_errors == floor(games$attack_errors), ]
stopifnot(!anyDuplicated(games[c("player_key", "team_key", "match_key")]),
          setequal(unique(games$division), c(1, 3)))
train <- games[games$match_date < as.Date("2025-01-01"), ]
test <- games[games$match_date >= as.Date("2025-01-01") &
              games$match_date < as.Date("2026-01-01"), ]
stopifnot(nrow(train) > 0, nrow(test) > 0,
          min(train$match_date) >= as.Date("2021-01-01"),
          !length(intersect(train$match_key, test$match_key)))

# Both models use the same outcome, input, and records.
linear_model <- lm(attack_errors ~ attack_attempts, data = train)
poisson_model <- glm(attack_errors ~ attack_attempts, data = train,
                     family = poisson(link = "log"))
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))
results <- data.frame(
  model = c("Linear", "Poisson"),
  intercept = c(unname(coef(linear_model)[1]), unname(coef(poisson_model)[1])),
  slope = c(unname(coef(linear_model)[2]), unname(coef(poisson_model)[2])),
  RMSE = c(rmse(test$attack_errors, predict(linear_model, test)),
           rmse(test$attack_errors, predict(poisson_model, test, type = "response")))
)
print(results, row.names = FALSE)
cat("Records:", nrow(games), "Training:", nrow(train), "Test:", nrow(test),
    "Excluded invalid counts:", n_input - nrow(games), "\n")
plot_max <- max(games$attack_attempts)
test_max <- max(test$attack_attempts)
examples <- data.frame(attack_attempts = c(0, 50, 75, 100, plot_max))
print(data.frame(examples, linear = predict(linear_model, examples),
                 poisson = predict(poisson_model, examples, type = "response")),
      row.names = FALSE)

# Show the full recorded attempt range and every valid test record.
grid <- data.frame(attack_attempts = seq(0, plot_max, length.out = 800))
linear_curve <- predict(linear_model, grid)
poisson_curve <- predict(poisson_model, grid, type = "response")
# Label the very large Poisson predictions above the displayed vertical scale.
display_max <- 40
above_axis <- data.frame(attack_attempts = c(100, plot_max))
above_axis$prediction <- predict(poisson_model, above_axis, type = "response")
draw_graph <- function() {
  old <- par(no.readonly = TRUE)
  on.exit(par(old), add = TRUE)
  par(mar = c(5.1, 5.1, 4.1, 1.2), las = 1, bty = "l", cex = 1.08)
  plot(grid$attack_attempts, linear_curve, type = "n", xaxt = "n",
       xaxs = "i", yaxs = "i", xlim = c(-2, plot_max + 3),
       ylim = c(-0.8, display_max),
       xlab = "Attack attempts in the same match", ylab = "Attack errors per match",
       main = "Attack errors: linear versus Poisson regression")
  axis(1, at = seq(0, plot_max, 25))
  abline(h = axTicks(2), col = "#EEEEEE")
  abline(v = test_max, col = "#AAAAAA", lty = 3)
  # Keep every match count at its true coordinates; transparency reveals overlap.
  points(test$attack_attempts, test$attack_errors, pch = 16, cex = 0.4,
         col = adjustcolor("#636363", alpha.f = 0.12))
  lines(grid$attack_attempts, linear_curve, col = "#2563A6", lwd = 3)
  lines(grid$attack_attempts, poisson_curve, col = "#C46B18", lwd = 3, lty = 2)
  text(test_max + 1.5, 27, paste("2025 observations end at", test_max),
       adj = c(0, 0.5), cex = 0.76, col = "#666666")
  labels <- paste0("At ", above_axis$attack_attempts, " attempts: ",
                   formatC(above_axis$prediction, format = "f", digits = 2, big.mark = ","))
  text(100, 36, paste(c("Poisson predictions above this axis", labels), collapse = "\n"),
       cex = 0.82, col = "#C46B18")
  edge_x <- (log(display_max) - coef(poisson_model)[1]) / coef(poisson_model)[2]
  arrow_x <- edge_x - 3
  arrow_y <- predict(poisson_model, data.frame(attack_attempts = arrow_x), type = "response")
  arrows(arrow_x, arrow_y, edge_x - 0.15, display_max - 0.35,
         length = 0.10, lwd = 2, col = "#C46B18")
  legend("topleft", bty = "n", cex = 0.9,
         legend = c(sprintf("Linear mean (RMSE %.3f)", results$RMSE[1]),
                    sprintf("Poisson mean (RMSE %.3f)", results$RMSE[2]),
                    "Individual 2025 match counts"),
         col = c("#2563A6", "#C46B18", "#999999"), lty = c(1, 2, NA),
         lwd = c(3, 3, NA), pch = c(NA, NA, 16), pt.cex = c(1, 1, 0.6))
  mtext(paste0("Training: 2021 to 2024 (n = ",
               format(nrow(train), big.mark = ",", trim = TRUE), "). Test: 2025 (n = ",
               format(nrow(test), big.mark = ",", trim = TRUE), ")"),
        side = 3, line = 0.5, cex = 0.85, col = "#555555")
}
graph <- file.path(folder, "graphs/mvb_linear_poisson_attack_errors.png")
dir.create(dirname(graph), recursive = TRUE, showWarnings = FALSE)
png(graph, width = 1600, height = 1050, res = 150)
tryCatch(draw_graph(), finally = dev.off())
if (interactive()) draw_graph()
cat("Graph:", graph, "\n")
saveRDS(list(results = results, linear_model = linear_model, poisson_model = poisson_model,
             n_train = nrow(train), n_test = nrow(test),
             n_plotted = nrow(test), plot_max = plot_max, test_max = test_max,
             above_axis = above_axis),
        file.path(folder, "data/mvb_attack_errors_linear_poisson.rds"))
