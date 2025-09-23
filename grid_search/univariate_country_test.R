library(data.table)
library(lme4)
library(car)
library(lmerTest)

# note this script was run using measles case data obtained April 4, 2025
# running it with refreshed data will likely yield different results.

working_dir <- "measles_forecasting/" # updated if needed!
set_wd(working_dir)
inputDat <- fread("./input/processed_measles_model_data.csv", na.strings = "")
setnames(inputDat, "outbreak_20_cuml_per_M", "outbreak")
inputDat[, date := as.Date(date)]
inputDat[inputDat[outbreak == "yes", .N, by = .(ISO3)], months_with_outbreak := i.N, on = .(ISO3)]
inputDat[is.na(months_with_outbreak), months_with_outbreak := 0]
inputDat[inputDat[outbreak == "yes", .N, by = .(ISO3, Year)][, .N, by = .(ISO3)], years_with_outbreak := i.N, on = .(ISO3)]
inputDat[is.na(years_with_outbreak), years_with_outbreak := 0]
outbreakDat <- unique(inputDat[, .(ISO3, months_with_outbreak, years_with_outbreak)])
inputDat[, outbreak := NULL]

reportDat <- inputDat[years_with_outbreak > 0, ]
# Prepare data for analysis
reportDat[, ISO3 := as.factor(ISO3)]
reportDat[outbreak_5_per_M == 1, outbreak := "yes"] 
reportDat[outbreak_5_per_M == 0, outbreak := "no"]
reportDat[, outbreak := as.factor(outbreak)]

predictors <- c(
  "birth_per_1k", "migrations_per_1k", 
  "MCV1", "MCV2",
  "passengers_from_iso3_reporting_cases", 
  "passengers_to_iso3_reporting_cases",
  "mnths_since_outbreak_20_cuml_per_M", 
  "mnths_since_outbreak_20_per_M", 
  "mnths_since_outbreak_2_per_M",
  "mnths_since_SIA",
  "outgoing_air_passengers", 
  "incoming_air_passengers",
  "prop_prev_rolling_12_mnths_outbreak_5M", 
  "prop_prev_rolling_24_mnths_outbreak_5M", 
  "prop_prev_rolling_60_mnths_outbreak_5M",
  "cases_1M_12z", 
  "cases_1M_36z", 
  "cases_1M_60z",
  "mean_temp",
  "mean_precip_mm_per_day"
)

# Create empty list to store results
results <- list()
model_errors <- list()

# Loop through each country
for (country in unique(reportDat$ISO3)) {
  # Subset data for this country
  country_data <- reportDat[ISO3 == country]
  
  # Create empty vector to store significant predictors and errors
  sig_predictors <- c()
  error_predictors <- c()
  
  # Loop through predictors
  for (pred in predictors) {
    # Initialize error flag
    has_error <- FALSE
    error_message <- NULL
    
    # Check for basic data issues
    if (sum(!is.na(country_data[[pred]])) < 3) {
      error_message <- "Too few non-NA observations"
      has_error <- TRUE
    } else if (is.na(var(country_data[[pred]], na.rm = TRUE)) || var(country_data[[pred]], na.rm = TRUE) == 0) {
      error_message <- "Zero or NA variance in predictor"
      has_error <- TRUE
    } else if (length(unique(country_data$outbreak[!is.na(country_data[[pred]])])) < 2) {
      error_message <- "Response variable has only one level after removing NA"
      has_error <- TRUE
    }
    
    if (!has_error) {
      # Fit logistic regression with error handling
      model_result <- tryCatch({
        model <- glm(outbreak ~ get(pred), 
                     data = country_data,
                     family = binomial(link = "logit"))
        
        # Check for convergence
        if (!model$converged) {
          list(error = TRUE, message = "Model failed to converge")
        } else {
          # Get p-value
          p_val <- summary(model)$coefficients[2,4]
          list(error = FALSE, p_value = p_val)
        }
      }, error = function(e) {
        list(error = TRUE, message = as.character(e))
      }, warning = function(w) {
        list(error = TRUE, message = as.character(w))
      })
      
      if (model_result$error) {
        has_error <- TRUE
        error_message <- model_result$message
      } else if (!is.na(model_result$p_value) && (model_result$p_value > 0.2 || pred %in% c("mean_temp", "mean_precip_mm_per_day"))) {
        sig_predictors <- c(sig_predictors, pred)
      }
    }
    
    # Store error information if there was an error
    if (has_error) {
      error_predictors <- c(error_predictors, 
                           sprintf("%s: %s", pred, error_message))
    }
  }
  
  # Store results for this country
  results[[as.character(country)]] <- sig_predictors
  if (length(error_predictors) > 0) {
    model_errors[[as.character(country)]] <- error_predictors
  }
}

# Print results
cat("\n=== PREDICTORS WITH P > 0.2 BY COUNTRY ===\n")
for (country in names(results)) {
  cat("\nCountry:", country, "\n")
  if (length(results[[country]]) > 0) {
    cat("Predictors with p > 0.2:", paste(results[[country]], collapse = ", "), "\n")
  } else {
    cat("No predictors with p > 0.2 found\n")
  }
}
# Create data frame for CSV output
csv_results <- data.frame(
  ISO3 = character(),
  predictor = character(), 
  p_value = numeric(),
  stringsAsFactors = FALSE
)

# Split data by country
split_data <- split(reportDat, reportDat$ISO3)

# Iterate through countries and predictors to build results
for (country in names(results)) {
  # Get predictors for this country
  country_predictors <- results[[country]]
  
  # Re-run models to get p-values
  country_data <- split_data[[country]]
  
  for (pred in predictors) {
    tryCatch({
      model <- glm(outbreak ~ get(pred),
                   data = country_data, 
                   family = binomial(link = "logit"))
      
      if (model$converged) {
        p_val <- summary(model)$coefficients[2,4]
        
        # Add row to results
        csv_results <- rbind(csv_results, data.frame(
          ISO3 = country,
          predictor = pred,
          p_value = p_val,
          stringsAsFactors = FALSE
        ))
      }
    }, error = function(e) {}, warning = function(w) {})
  }
}


csv_results <- data.table(csv_results)
csv_results[, .N, by = .(ISO3)]# Write results to CSV
write.csv(csv_results, "univariate_country_results.csv", row.names = FALSE)

# Print errors if any occurred
if (length(model_errors) > 0) {
  cat("\n=== MODEL FITTING ERRORS ===\n")
  for (country in names(model_errors)) {
    cat("\nCountry:", country, "\n")
    cat("Errors:\n")
    cat(paste("  -", model_errors[[country]], collapse = "\n"), "\n")
  }
}
