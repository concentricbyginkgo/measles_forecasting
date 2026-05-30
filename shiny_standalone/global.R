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
# Contact: Amanda Meadows ~ amanda.meadows612@gmail.com 
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
# Try to load compiled summaries (both selection and validation), fallback to sample data
# If both selection and validation summaries exist, combine them
run_name <- "test_run"  # Update this to match your run name
selection_summary <- paste0("../model/output/", run_name, "_selection_compiled_summary.csv")
validation_summary <- paste0("../model/output/", run_name, "_validation_compiled_summary.csv")
generic_summary <- paste0("../model/output/", run_name, "_compiled_summary.csv")

summary_list <- list()
if (file.exists(selection_summary)) {
  summary_list[["selection"]] <- fread(selection_summary, na.strings = "")
  message("Loaded selection compiled summary table")
}
if (file.exists(validation_summary)) {
  summary_list[["validation"]] <- fread(validation_summary, na.strings = "")
  message("Loaded validation compiled summary table")
}

if (length(summary_list) > 0) {
  # Combine selection and validation summaries if both exist
  summaryTable <- rbindlist(summary_list, fill = TRUE)
  message(paste("Combined", length(summary_list), "summary table(s) from model output"))
} else if (file.exists(generic_summary)) {
  # Fallback to generic compiled summary (for backward compatibility)
  summaryTable <- fread(generic_summary, na.strings = "")
  message("Loaded generic compiled summary table from model output")
} else {
  summaryTable <- fread("data/sample_summaryTable.csv", na.strings = "")
  message("Loaded sample summary table (compiled summaries not found)")
}

# Process summary table: convert numeric columns and round to 3 decimal places
summaryTable[, (num_cols) := lapply(.SD, as.numeric), .SDcols = num_cols]
summaryTable[, (num_cols) := lapply(.SD, function(x) round(x, 3)), .SDcols = num_cols]

# Select relevant columns for the application (handle missing columns gracefully)
table_cols <- c("ID", "MODEL_ID", "geography", "predictor", "model", num_cols)
available_table_cols <- intersect(table_cols, names(summaryTable))
examineDat <- summaryTable[, ..available_table_cols]

# Create country selection list with ISO3 codes and full country names
country_list <- sort(summaryTable[, unique(ID)])
names(country_list) <- paste0(country_list, " - ", countrycode::countrycode(country_list, 'iso3c', 'country.name'))
country_list <- c(country_list)

# Load cutoff dates for model selection and validation periods
cutoff_dat <- fread("data/cutoff_date_by_country.csv")
if (!("GEO_ID" %in% names(cutoff_dat)) && "ISO3" %in% names(cutoff_dat)) {
  setnames(cutoff_dat, "ISO3", "GEO_ID")
}
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
      # Handle missing run or is_cluster_run columns
      if ("run" %in% names(summ_dt) && "is_cluster_run" %in% names(summ_dt)) {
        out_summ <- summ_dt[ID == iso3,  head(.SD, n), by = .(run, is_cluster_run)]
      } else {
        out_summ <- summ_dt[ID == iso3,  head(.SD, n)]
      }
    }
  }
  
  # Add ranking based on the specified column
  out_summ[, rank := frank(get(col_name), ties.method = "dense")]
  
  # Process selection period time series data
  # Merge on ROW_ID if available, otherwise try MODEL_ID
  if ("ROW_ID" %in% names(s_iso3_tables) && "ROW_ID" %in% names(out_summ)) {
    s_out_tables <- s_iso3_tables[out_summ, .SD, on = .(ID, ROW_ID)]
  } else if ("MODEL_ID" %in% names(s_iso3_tables) && "MODEL_ID" %in% names(out_summ)) {
    s_out_tables <- s_iso3_tables[out_summ, .SD, on = .(ID, MODEL_ID)]
  } else {
    s_out_tables <- s_iso3_tables[out_summ, .SD, on = .(ID)]
  }
  
  # Add columns from summary table if they exist
  if ("is_cluster_run" %in% names(out_summ)) {
    s_out_tables[out_summ, is_cluster_run := i.is_cluster_run, on = .(ID, ROW_ID = ROW_ID)]
  } else {
    s_out_tables[, is_cluster_run := "no"]
  }
  if ("MODEL_ID" %in% names(out_summ) && !"MODEL_ID" %in% names(s_out_tables)) {
    s_out_tables[out_summ, MODEL_ID := i.MODEL_ID, on = .(ID, ROW_ID = ROW_ID)]
  }
  if ("model" %in% names(out_summ)) {
    s_out_tables[out_summ, model := i.model, on = .(ID, ROW_ID = ROW_ID)]
  }
  s_out_tables[cutoff_dat, cutoff_date := i.s_cutoff_date, on = .(ID = GEO_ID)]
  s_out_tables[cutoff_dat, end_date := i.s_end_date, on = .(ID = GEO_ID)]
  s_out_tables[, run_period := "selection"]
  
  # Process validation period data if available
  if(nrow(v_iso3_tables)>0){
    
    # Get validation time series tables
    if ("MODEL_ID" %in% names(v_iso3_tables) && "MODEL_ID" %in% names(out_summ)) {
      v_out_tables <- v_iso3_tables[out_summ, .SD, on = .(ID, MODEL_ID)]
    } else {
      v_out_tables <- v_iso3_tables[out_summ, .SD, on = .(ID)]
    }
    
    # Add columns from summary table if they exist
    if ("is_cluster_run" %in% names(out_summ)) {
      v_out_tables[out_summ, is_cluster_run := i.is_cluster_run, on = .(ID, MODEL_ID = MODEL_ID)]
    } else {
      v_out_tables[, is_cluster_run := "no"]
    }
    if ("model" %in% names(out_summ)) {
      v_out_tables[out_summ, model := i.model, on = .(ID, MODEL_ID = MODEL_ID)]
    }
    v_out_tables[cutoff_dat, cutoff_date := i.v_cutoff_date, on = .(ID = GEO_ID)]
    v_out_tables[cutoff_dat, end_date := i.v_end_date, on = .(ID = GEO_ID)]
    v_out_tables[, run_period := "validation"]
    
    # Combine selection and validation data
    out_tables <- rbind(s_out_tables,
                        v_out_tables)
  }else{
    out_tables <- s_out_tables
  }
  
  # Add ranking and format dates
  if ("ROW_ID" %in% names(out_tables) && "ROW_ID" %in% names(out_summ)) {
    out_tables[out_summ, rank := i.rank, on = .(ID, ROW_ID)]
  } else if ("MODEL_ID" %in% names(out_tables) && "MODEL_ID" %in% names(out_summ)) {
    out_tables[out_summ, rank := i.rank, on = .(ID, MODEL_ID)]
  }
  out_tables[, ds := as.Date(ds)]
  
  # Format MODEL_ID if model column exists
  if ("model" %in% names(out_tables) && "MODEL_ID" %in% names(out_tables)) {
    out_tables[, MODEL_ID := ifelse(model %in% c("boosted heavy", "diverse"), paste0("E-", MODEL_ID), paste0("S-", MODEL_ID))]
    out_tables[, model := NULL]
  }
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

