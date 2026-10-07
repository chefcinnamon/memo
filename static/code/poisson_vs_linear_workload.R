# Predict attack errors from earlier attacking workload, with one shared input.
# Source: https://jeffreyrstevens.github.io/ncaavolleyballr/articles/data.html
library(DBI)
library(RSQLite)
folder <- path.expand("~/Desktop/volleyball-projections")
database <- file.path(folder, "data/mens_regression_comparison.sqlite")
if (!file.exists(database)) stop("The men's regression database is missing: ", database)
con <- dbConnect(SQLite(), database, flags = SQLITE_RO)
games <- tryCatch(dbGetQuery(con, "
  WITH history AS (
    SELECT *, COUNT(*) OVER earlier AS prior_matches,
      SUM(attack_attempts) OVER earlier AS prior_attempts,
      MAX(match_date) OVER earlier AS latest_prior_date
    FROM player_match_stats
    WHERE attack_attempts >= 0 AND attack_errors >= 0
      AND attack_errors <= attack_attempts
    WINDOW earlier AS (
      PARTITION BY division, archive_year, team_key, player_key
      ORDER BY match_date GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
  )
  SELECT division, match_key, team_key, player_key, match_date, latest_prior_date,
    prior_matches, prior_attempts * 1.0 / prior_matches AS prior_average_attempts,
    attack_errors
  FROM history WHERE prior_matches >= 3
"), finally = dbDisconnect(con))
games$match_date <- as.Date(games$match_date)
games$latest_prior_date <- as.Date(games$latest_prior_date)
stopifnot(all(complete.cases(games)),
          all(games$latest_prior_date < games$match_date),
          !anyDuplicated(games[c("match_key", "team_key", "player_key")]))
# log(1 + average) keeps zero workloads defined without inventing observations.
games$logged_workload <- log1p(games$prior_average_attempts)
train <- games[games$match_date < as.Date("2025-01-01"), ]
test <- games[games$match_date >= as.Date("2025-01-01") &
              games$match_date < as.Date("2026-01-01"), ]
stopifnot(nrow(train) > 0, nrow(test) > 0,
          !length(intersect(train$match_key, test$match_key)))
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))

# This feature was chosen using 2024 validation, before evaluating 2025.
development <- train[train$match_date < as.Date("2024-01-01"), ]
validation <- train[train$match_date >= as.Date("2024-01-01"), ]
validation_linear <- lm(attack_errors ~ logged_workload, data = development)
validation_poisson <- glm(attack_errors ~ logged_workload, data = development,
                          family = poisson(link = "log"))
cat("Validation RMSE:",
    rmse(validation$attack_errors, predict(validation_linear, validation)),
    rmse(validation$attack_errors, predict(validation_poisson, validation, type = "response")),
    "\n")

# Both final models use the same feature and training records.
linear_model <- lm(attack_errors ~ logged_workload, data = train)
poisson_model <- glm(attack_errors ~ logged_workload, data = train,
                     family = poisson(link = "log"))
results <- data.frame(
  model = c("Linear", "Poisson"),
  intercept = c(unname(coef(linear_model)[1]), unname(coef(poisson_model)[1])),
  slope = c(unname(coef(linear_model)[2]), unname(coef(poisson_model)[2])),
  RMSE = c(rmse(test$attack_errors, predict(linear_model, test)),
           rmse(test$attack_errors, predict(poisson_model, test, type = "response")))
)
print(results, row.names = FALSE)
improvement <- 100 * (1 - results$RMSE[2] / results$RMSE[1])
dispersion <- sum(residuals(poisson_model, type = "pearson")^2) / df.residual(poisson_model)
cat("Training:", nrow(train), "Test:", nrow(test),
    "Poisson RMSE reduction:", improvement, "percent\n")
cat("Poisson residual dispersion:", dispersion, "\n")
examples <- data.frame(prior_average_attempts = c(0, 5, 20, 40))
examples$logged_workload <- log1p(examples$prior_average_attempts)
print(data.frame(examples, linear = predict(linear_model, examples),
                 poisson = predict(poisson_model, examples, type = "response")),
      row.names = FALSE)

grid <- data.frame(logged_workload = seq(0, max(test$logged_workload), length.out = 400))
linear_curve <- predict(linear_model, grid)
poisson_curve <- predict(poisson_model, grid, type = "response")
draw_graph <- function() {
  old <- par(no.readonly = TRUE)
  on.exit(par(old), add = TRUE)
  par(mar = c(5.1, 5.1, 4.1, 1.2), las = 1, bty = "l", cex = 1.08)
  low <- min(0, linear_curve) - 0.15
  high <- max(linear_curve, poisson_curve, test$attack_errors) * 1.08
  plot(grid$logged_workload, linear_curve, type = "n", xaxs = "i", yaxs = "i",
       xlim = c(-0.05, max(grid$logged_workload) + 0.05), ylim = c(low, high),
       xlab = "log(1 + earlier average attack attempts)", ylab = "Attack errors per match",
       main = "Earlier workload: linear versus Poisson regression")
  rect(par("usr")[1], low, par("usr")[2], 0, col = "#FAF3F3", border = NA)
  abline(h = axTicks(2), col = "#EEEEEE")
  abline(h = 0, col = "#AAAAAA", lty = 3)
  # Plot every 2025 record at its true coordinates. Transparency reveals overlap.
  points(test$logged_workload, test$attack_errors, pch = 16, cex = 0.4,
         col = adjustcolor("#636363", alpha.f = 0.12))
  lines(grid$logged_workload, linear_curve, col = "#2563A6", lwd = 3)
  lines(grid$logged_workload, poisson_curve, col = "#C46B18", lwd = 3, lty = 2)
  legend("topleft", bty = "n", cex = 0.9,
         legend = c(sprintf("Linear mean (RMSE %.3f)", results$RMSE[1]),
                    sprintf("Poisson mean (RMSE %.3f)", results$RMSE[2]),
                    "Individual 2025 match counts"),
         col = c("#2563A6", "#C46B18", "#999999"),
         lty = c(1, 2, NA), lwd = c(3, 3, NA),
         pch = c(NA, NA, 16), pt.cex = c(1, 1, 0.6))
  mtext(paste0("Training: 2021 to 2024 (n = ", format(nrow(train), big.mark = ",", trim = TRUE),
               "). Test: 2025 (n = ", format(nrow(test), big.mark = ",", trim = TRUE), ")"),
        side = 3, line = 0.5, cex = 0.85, col = "#555555")
}
graph <- file.path(folder, "graphs/mvb_linear_poisson_workload.png")
dir.create(dirname(graph), recursive = TRUE, showWarnings = FALSE)
png(graph, width = 1600, height = 1050, res = 150)
tryCatch(draw_graph(), finally = dev.off())
if (interactive()) draw_graph()
cat("Graph:", graph, "\n")
saveRDS(list(results = results, linear_model = linear_model, poisson_model = poisson_model,
             n_train = nrow(train), n_test = nrow(test),
             improvement = improvement, dispersion = dispersion),
        file.path(folder, "data/mvb_workload_linear_poisson.rds"))
