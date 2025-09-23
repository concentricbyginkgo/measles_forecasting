library(data.table)
library(digest)
outfile_name <- "run_metadata.csv"
out_path <- "~/measles_forecasting/grid_search/"
best_pred_by_country <- fread("univariate_country_results.csv")
best_pred_by_country <- best_pred_by_country[!(predictor %in% c("passengers_from_iso3_reporting_cases", "passengers_to_iso3_reporting_cases"))]

correlated_vars <- fread("correlation_results.csv")
correlated_vars <- correlated_vars[p_value<0.2 & correlation>0.7]

# Create separate table for climate variables
climate_vars <- c("mean_temp", "mean_precip_mm_per_day")
climate_by_country <- best_pred_by_country[predictor %in% climate_vars]

# Filter main predictors table to exclude climate and keep only significant predictors
best_pred_by_country <- best_pred_by_country[!(predictor %in% climate_vars)]
best_pred_by_country <- best_pred_by_country[p_value <= 0.2]

# Function to check if predictors are similar
are_similar_predictors <- function(pred1, pred2) {
  # Define groups of similar predictors
  similar_groups <- list(
    c("cases_1M_12z", "cases_1M_36z", "cases_1M_60z"),
    c("prop_prev_rolling_12_mnths_outbreak_5M", "prop_prev_rolling_24_mnths_outbreak_5M", "prop_prev_rolling_60_mnths_outbreak_5M"),
    c("mnths_since_outbreak_20_cuml_per_M", "mnths_since_outbreak_20_per_M", "mnths_since_outbreak_2_per_M")
  )
  
  # Check if predictors are in same group
  for(group in similar_groups) {
    if(pred1 %in% group && pred2 %in% group) return(TRUE)
  }
  return(FALSE)
}

# Function to check if any predictors in combination are similar
has_similar_predictors <- function(pred_combo) {
  if(length(pred_combo) < 2) return(FALSE)
  for(i in 1:(length(pred_combo)-1)) {
    for(j in (i+1):length(pred_combo)) {
      if(are_similar_predictors(pred_combo[i], pred_combo[j])) return(TRUE)
    }
  }
  return(FALSE)
}

# Function to check if predictors are correlated for a given country
has_correlated_predictors <- function(pred_combo, country, corr_data) {
  if(length(pred_combo) < 2) return(FALSE)
  for(i in 1:(length(pred_combo)-1)) {
    for(j in (i+1):length(pred_combo)) {
      corr_pair <- corr_data[ISO3 == country & 
                               ((predictor1 == pred_combo[i] & predictor2 == pred_combo[j]) |
                                  (predictor1 == pred_combo[j] & predictor2 == pred_combo[i]))]
      if(nrow(corr_pair) > 0) return(TRUE)
    }
  }
  return(FALSE)
}

# Create combinations for each country
predictor_combinations <- data.table()

for(country in unique(best_pred_by_country$ISO3)) {
  # Get predictors for this country
  country_preds <- unique(best_pred_by_country[ISO3 == country]$predictor)
  
  # Get climate variables for this country
  country_climate <- climate_by_country[ISO3 == country, predictor]
  climate_string <- ifelse(length(country_climate) > 0, paste(country_climate, collapse = "|"), NA)
  
  # Generate all combinations of 1, 2, and 3 predictors
  for(n in 1:min(3, length(country_preds))) {
    combos <- combn(country_preds, n, simplify = FALSE)
    
    # Filter out combinations with similar predictors or correlated predictors
    valid_combos <- combos[!sapply(combos, has_similar_predictors) & 
                             !sapply(combos, function(x) has_correlated_predictors(x, country, correlated_vars))]
    
    # Add to results
    if(length(valid_combos) > 0) {
      predictor_combinations <- rbindlist(
        list(
          predictor_combinations,
          data.table(
            ISO3 = country,
            predictor = sapply(valid_combos, paste, collapse = "|"),
            environmentalArgs = climate_string
          )
        )
      )
    }
  }
}
predictor_combinations

models <- c("Random Forest",
            "Bagging regressor", 
            "CatBoost",
            "gradient boosting",
            "XGBoost",
            "boosted heavy",
            "diverse")
#"boosted heavy",
#"diverse")
# Create all combinations of countries, predictor sets and models
final_combinations <- CJ(
  ISO3 = unique(predictor_combinations$ISO3),
  predictor = unique(predictor_combinations$predictor),
  model = models,
  Rep = 1:15
)

# Only keep valid country-predictor combinations that exist in predictor_combinations
final_combinations <- final_combinations[
  predictor_combinations, 
  on = .(ISO3, predictor),
  nomatch = NULL
]
final_combinations[, environmentalArgs := "mean_temp|mean_precip_mm_per_day"]


predictor_strings <- unique(predictor_combinations$predictor)
subset_strings <- predictor_strings[sapply(strsplit(predictor_strings, "\\|"), length) <= 2]

# Create all combinations of clusters, predictor sets and models
cluster_combinations <- CJ(
  ISO3 = paste0("cluster:", 1:10),
  predictor = subset_strings,
  environmentalArgs = "mean_temp|mean_precip_mm_per_day",
  model = models,
  Rep = 1:15
)
cluster_combinations[, .N, by = .(predictor)]
final_combinations[, .N, by = .(predictor)]
out_combinations <- rbind(cluster_combinations, final_combinations)
# Convert predictor format to JSON
out_combinations[, predictor_json := paste0("{'", gsub("\\|", "': 0, '", predictor), "': 0}")]
out_combinations[, num_predictors := lengths(strsplit(predictor, "\\|"))]

# Convert environmental args to JSON format  
out_combinations[, environmentalArg := "{'mean_temp': 3, 'mean_precip_mm_per_day': 3}"]
out_combinations
# Generate MODEL_ID (using first 12 characters of a hash)
out_combinations[, MODEL_ID := substr(digest(paste(country, predictor, model), algo = "md5"), 1, 12), 
                 by = .(country, predictor, model)]
out_combinations[, .N, by = .(MODEL_ID)][N!=15]
# Add row numbers for unique model IDs
out_combinations[, ROW_ID := .I]
out_combinations[, Seed := sample(1:2000000, .N)]
out_combinations[, .N, by = .(Seed)][N>1]

# Reorder and select columns to match target format
setnames(out_combinations, "ISO3", "country")
final_output <- out_combinations[, .(MODEL_ID, country, predictor = predictor_json, num_predictors, environmentalArg, model, Rep, ROW_ID, Seed)]
final_output[, .N, by = .(MODEL_ID)]
fwrite(final_output, paste0(outpath, outfile_name))

