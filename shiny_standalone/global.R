############################################
# MEASLES MODEL VALIDATION SHINY APP - GLOBAL SCRIPT
# 
# This script contains global variables, functions, and data loading
# for a Shiny application that visualizes measles model selection and validation results.
# 
# The app allows users to:
# - Select countries and view model performance metrics
# - Visualize epidemiological curves for model selection and validation periods
# - Compare binary outbreak predictions vs observed data
# - Explore model rankings based on various performance metrics
#
# Author: Ginkgo Biosecurity
# Date: September 17, 2025
# Version: 1.0
############################################

# Enable Shiny reactivity logging for debugging
options(shiny.reactlog = TRUE)

# Load required packages
library(shiny)        # Web application framework
library(data.table)   # Fast data manipulation and file I/O
library(plotly)       # Interactive plotting
library(ggplot2)      # Static plotting
library(DT)           # Data tables for Shiny
library(viridis)      # Color palettes for plots

# Disable scientific notation for better readability
options(scipen = 999)

# Define numeric columns for model performance metrics
# These columns contain continuous values that will be used for ranking and display
num_cols <- c("Combine_RMSE5", "v_Combine_RMSE5",
              "Combine_MSE5","v_Combine_MSE5", 
              "Combine_MAE5", "v_Combine_MAE5",
              "Test_MSE", "v_Test_MSE",
              "Test_MAE", "v_Test_MAE", 
              "Test_R2","v_Test_R2",
              "Train_MSE","v_Train_MSE", 
              "Train_MAE", "v_Train_MAE",
              "Train_R2", "v_Train_R2")

# Load main summary table from file system containing model performance metrics
summaryTable <- fread("data/sample_summaryTable.csv", na.strings = "")

# Process summary table: convert numeric columns and round to 3 decimal places
summaryTable[, (num_cols) := lapply(.SD, as.numeric), .SDcols = num_cols]
summaryTable[, (num_cols) := lapply(.SD, function(x) round(x, 3)), .SDcols = num_cols]

# Select relevant columns for the application
table_cols <- c("ID", "MODEL_ID", "country", "predictor", "model", "Prediction_Tier", "v_Prediction_Tier", num_cols)
examineDat <- summaryTable[, ..table_cols]

# Create country selection list with ISO3 codes and full country names
country_list <- sort(summaryTable[, unique(ID)])
names(country_list) <- paste0(country_list, " - ", countrycode::countrycode(country_list, 'iso3c', 'country.name'))
country_list <- c(country_list)

# Load cutoff dates for model selection and validation periods
cutoff_dat <- fread("data/cutoff_date_by_country.csv")
cutoff_dat[, v_cutoff_date := as.Date(cutoff_date)]
cutoff_dat[, v_end_date := lubridate::add_with_rollback(v_cutoff_date, months(9))]
cutoff_dat[, s_cutoff_date := as.Date(selection_cutoff_date)]
cutoff_dat[, s_end_date := lubridate::add_with_rollback(s_cutoff_date, months(9))]

#' Get Plot Data for Model Visualization
#' 
#' This function retrieves and processes time series data for model selection and validation
#' periods for a specific country. It loads data from the file system, ranks models by performance,
#' and combines selection and validation datasets.
#' 
#' @param summ_dt data.table containing model summary statistics
#' @param iso3 character string of ISO3 country code
#' @param col_name character string of column name to rank models by
#' @param n integer, number of top models to return (NULL for all)
#' @param cutoff_dat data.table containing cutoff dates for selection/validation periods
#' @param by_config character string, if "yes" groups by run and is_cluster_run
#' @return data.table containing time series data with model rankings and period labels
get_plot_dat <-function(summ_dt, iso3, col_name, n = NULL, cutoff_dat, by_config = NULL){
  
  # Load model selection time series data for the country
  s_iso3_tables <- fread(paste0("data/tables/selection/", iso3, ".csv"))
  
  # Try to load validation data (may not exist for all countries)
  tryCatch({
    v_iso3_tables <- fread(paste0("data/tables/validation/", iso3, ".csv"))
  }, error = function(e) {
    msg <- paste0("No validation runs completed for ", iso3)
    print(msg)
    v_iso3_tables <- NULL
  })
  
  # Sort summary data by the specified column and filter for the country
  summ_dt <- summ_dt[order(get(col_name))]
  if(is.null(n)){
    out_summ <- summ_dt[ID == iso3]
  }else{
    if(is.null(by_config)){
      out_summ <- summ_dt[ID == iso3,  head(.SD, n)]
    }else{
      out_summ <- summ_dt[ID == iso3,  head(.SD, n), by = .(run, is_cluster_run)]
    }
  }
  
  # Add ranking based on the specified column
  out_summ[, rank := frank(get(col_name), ties.method = "dense")]
  
  # Process selection period time series data
  s_out_tables <- s_iso3_tables[out_summ, .SD, on = .(ID, ROW_ID)]
  s_out_tables[out_summ, `:=`(is_cluster_run = i.is_cluster_run,
                              MODEL_ID = i.MODEL_ID,
                              model = i.model), on = .(ID, ROW_ID)]
  s_out_tables[cutoff_dat, cutoff_date := i.s_cutoff_date, on = .(ID = ISO3)]
  s_out_tables[cutoff_dat, end_date := i.s_end_date, on = .(ID = ISO3)]
  s_out_tables[, run_period := "selection"]
  
  # Process validation period data if available
  if(nrow(v_iso3_tables)>0){
    
    # Get validation time series tables
    v_out_tables <- v_iso3_tables[out_summ, .SD, on = .(ID, MODEL_ID)]
    v_out_tables[out_summ, `:=`(is_cluster_run = i.is_cluster_run,
                                model = i.model), on = .(ID, MODEL_ID)]
    v_out_tables[cutoff_dat, cutoff_date := i.v_cutoff_date, on = .(ID = ISO3)]
    v_out_tables[cutoff_dat, end_date := i.v_end_date, on = .(ID = ISO3)]
    v_out_tables[, run_period := "validation"]
    
    # Combine selection and validation data
    out_tables <- rbind(s_out_tables,
                        v_out_tables)
  }else{
    out_tables <- s_out_tables
  }
  
  # Add ranking and format dates
  out_tables[out_summ, rank := i.rank, on = .(ID, ROW_ID)]
  out_tables[, ds := as.Date(ds)]
  out_tables[, MODEL_ID := ifelse(model %in% c("boosted heavy", "diverse"), paste0("E-", MODEL_ID), paste0("S-", MODEL_ID))]
  out_tables[, model := NULL]
  return(out_tables)
}

#' Plot Binary Outcome Predictions
#' 
#' Creates a heatmap visualization comparing predicted vs observed binary outbreak outcomes
#' (5M threshold) for the top-ranked models. Shows model performance over time with
#' vertical lines indicating cutoff and end dates for the evaluation period.
#' 
#' @param plot_dat data.table containing time series data with model predictions
#' @param period character string, either "selection" or "validation" 
#' @return ggplot object or plotly empty plot if no data available
plot_binary_outcome <- function(plot_dat, period){
  
  # Filter data for the specified period
  top_iso3_plot_dat <- plot_dat[run_period == period]
  
  if(nrow(top_iso3_plot_dat) > 0){
    # Sort by model ID and date, convert dates to character for plotting
    top_iso3_plot_dat <- top_iso3_plot_dat[order(MODEL_ID, ds)]
    top_iso3_plot_dat[, char_date := as.character(ds)]
    
    # Create observed data row
    obs_outbreak_dat <- unique(top_iso3_plot_dat[, .(ds, char_date, outbreak_5M = outbreak_observed_5M, rank = 0)])
    obs_outbreak_dat[, MODEL_ID := "Observed"]
    
    # Combine predicted and observed data
    binary_dat <- rbind(top_iso3_plot_dat[, .(MODEL_ID, ds, char_date, outbreak_5M = outbreak_predicted_5M, rank)],
                        obs_outbreak_dat)
    
    # Filter to last 5 years of data
    binary_dat <- binary_dat[ds >= max(ds) - lubridate::years(5)]
    ordered_rows <- unique(binary_dat[order(-rank, ds), MODEL_ID])
    
    # Convert to factors for proper plotting
    binary_dat[, outbreak_5M := factor(outbreak_5M, levels = c("no", "yes"))]
    binary_dat[, MODEL_ID := factor(MODEL_ID, levels = ordered_rows, ordered = TRUE)]
    
    # Calculate positions for cutoff and end date lines
    binary_start_date <- as.character(unique(top_iso3_plot_dat$cutoff_date))
    start_position <- match(binary_start_date, unique(binary_dat$char_date))
    binary_end_date <- as.character(unique(top_iso3_plot_dat$end_date))
    end_position <- match(binary_end_date, unique(binary_dat$char_date))
    
    # Create heatmap plot
    p <- ggplot(binary_dat) + 
      geom_tile(aes(x = char_date, y = MODEL_ID, fill = outbreak_5M), color = "black") +
      scale_fill_manual(name = "Outbreak 5M", values = c("#4258A5","#099079"),
                        limits = c("no", "yes"),                             
                        drop = FALSE)+
      geom_vline(xintercept = start_position - 0.5, color = "firebrick") +
      geom_vline(xintercept = end_position - 0.5, color = "firebrick") +
      scale_x_discrete(limits=unique(binary_dat$char_date), breaks=unique(binary_dat$char_date)[seq(1,length(unique(binary_dat$char_date)),by=12)])+
      xlab("")+
      theme(axis.text.x = element_text(angle = 70, hjust=1))
    
  }else{
    # Return empty plot with message if no data available
    print("No runs available to plot")
    p <- plotly::plot_ly() %>%
      plotly::layout(
        xaxis = list(showticklabels = FALSE),
        yaxis = list(showticklabels = FALSE),
        annotations = list(
          text = "No runs available to plot",
          x = 0.5, y = 0.5,
          xref = "paper", yref = "paper",
          showarrow = FALSE,
          font = list(size = 16)
        )
      )
  }
  return(p)
}

