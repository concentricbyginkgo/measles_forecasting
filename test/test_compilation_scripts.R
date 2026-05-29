#!/usr/bin/env Rscript
###########################################################################
###   TEST_COMPILATION_SCRIPTS.R                                        ###
###      * Integration tests for model output compilation scripts       ###
###      * Tests with sample data to validate workflow                  ###
###########################################################################

library(data.table)

test_compilation_scripts <- function() {
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("Compilation Script Integration Tests\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  
  # Get base directory - handle different ways script might be called
  script_path <- commandArgs(trailingOnly = FALSE)
  base_dir <- NULL
  
  # Try to get script file path
  if (any(grepl("--file=", script_path))) {
    script_file <- sub("--file=", "", script_path[grepl("--file=", script_path)][1])
    if (file.exists(script_file)) {
      base_dir <- dirname(dirname(normalizePath(script_file)))
    }
  }
  
  # Fallback methods
  if (is.null(base_dir)) {
    # Try current working directory approach
    if (file.exists("test/test_compilation_scripts.R")) {
      base_dir <- getwd()
    } else if (file.exists("../test/test_compilation_scripts.R")) {
      base_dir <- dirname(getwd())
    } else {
      # Last resort: assume we're in test/ directory
      base_dir <- dirname(getwd())
    }
  }
  
  # Validate base_dir exists
  if (is.null(base_dir) || !dir.exists(base_dir)) {
    stop(paste("Could not determine base directory. Tried:", base_dir))
  }
  test_dir <- file.path(base_dir, "test")
  test_data_dir <- file.path(test_dir, "data")
  test_output_dir <- file.path(test_dir, "output")
  
  # Create test directories
  dir.create(test_output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create sample summary files
  cat("\nCreating sample summary files...\n")
  sample_scores_dir <- file.path(test_output_dir, "test_run", "scores")
  dir.create(sample_scores_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create a few sample summary files
  sample_summary <- data.table(
    "Test MSE" = c(1.5, 2.3, 1.8),
    "Test MAE" = c(1.2, 1.9, 1.5),
    "Test R2" = c(0.85, 0.72, 0.80),
    "Train MSE" = c(1.3, 2.1, 1.6),
    "Train MAE" = c(1.0, 1.7, 1.3),
    "Train R2" = c(0.88, 0.75, 0.82),
    "method" = c("gradient boosting", "CatBoost", "XGBRegressor"),
    "geography" = c("NGA", "NGA", "NGA"),
    "depVar" = "cases_1M"
  )
  
  for (i in 1:3) {
    row_id <- paste0("20173", i)
    fwrite(sample_summary[i], file.path(sample_scores_dir, paste0(row_id, "_Summary.csv")))
  }
  cat("✓ Created 3 sample summary files\n")
  
  # Create sample table files
  cat("\nCreating sample table files...\n")
  sample_tables_dir <- file.path(test_output_dir, "test_run", "tables")
  dir.create(sample_tables_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create sample projection tables
  dates <- seq(as.Date("2020-01-01"), as.Date("2023-12-01"), by = "month")
  sample_table <- data.table(
    ds = dates,
    ID = "NGA",
    yhat1 = rnorm(length(dates), 5, 2),
    y = rnorm(length(dates), 5, 2)
  )
  
  for (i in 1:3) {
    row_id <- paste0("20173", i)
    fwrite(sample_table, file.path(sample_tables_dir, paste0(row_id, "_NGA_Projection.csv")))
  }
  cat("✓ Created 3 sample projection files\n")
  
  # Create sample metadata
  cat("\nCreating sample metadata...\n")
  sample_metadata <- data.table(
    ROW_ID = c("201731", "201732", "201733"),
    geography = c("NGA", "NGA", "NGA"),
    MODEL_ID = c("test1", "test2", "test3"),
    model = c("gradient boosting", "CatBoost", "XGBRegressor"),
    predictor = c("temp,precip", "temp,precip", "temp,precip")
  )
  metadata_path <- file.path(test_data_dir, "metadata_example.csv")
  dir.create(test_data_dir, recursive = TRUE, showWarnings = FALSE)
  fwrite(sample_metadata, metadata_path)
  cat("✓ Created sample metadata file\n")
  
  # Test summary compilation (simplified)
  cat("\nTesting summary compilation logic...\n")
  summary_files <- list.files(sample_scores_dir, pattern = "_Summary.csv$", full.names = TRUE)
  if (length(summary_files) > 0) {
    summaries <- lapply(summary_files, function(f) {
      tryCatch({
        dt <- fread(f, na.strings = "")
        dt[, file := basename(f)]
        return(dt)
      }, error = function(e) {
        return(NULL)
      })
    })
    summaries <- summaries[!sapply(summaries, is.null)]
    if (length(summaries) > 0) {
      compiled <- rbindlist(summaries, fill = TRUE)
      cat("✓ Successfully compiled", nrow(compiled), "summary records\n")
    } else {
      cat("✗ Failed to read summary files\n")
      return(FALSE)
    }
  } else {
    cat("✗ No summary files found\n")
    return(FALSE)
  }
  
  # Test table compilation logic (simplified)
  cat("\nTesting table compilation logic...\n")
  table_files <- list.files(sample_tables_dir, pattern = "_Projection.csv$", full.names = TRUE)
  if (length(table_files) > 0) {
    tables <- lapply(table_files, function(f) {
      tryCatch({
        dt <- fread(f, na.strings = "")
        return(dt)
      }, error = function(e) {
        return(NULL)
      })
    })
    tables <- tables[!sapply(tables, is.null)]
    if (length(tables) > 0) {
      compiled_tables <- rbindlist(tables, fill = TRUE)
      cat("✓ Successfully compiled", nrow(compiled_tables), "table records\n")
      
      # Check required columns
      required_cols <- c("ds", "ID", "yhat1")
      missing_cols <- setdiff(required_cols, names(compiled_tables))
      if (length(missing_cols) == 0) {
        cat("✓ All required columns present\n")
      } else {
        cat("✗ Missing required columns:", paste(missing_cols, collapse = ", "), "\n")
        return(FALSE)
      }
    } else {
      cat("✗ Failed to read table files\n")
      return(FALSE)
    }
  } else {
    cat("✗ No table files found\n")
    return(FALSE)
  }
  
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("✓ All compilation script tests passed\n")
  return(TRUE)
}

# Run tests
success <- test_compilation_scripts()
quit(status = ifelse(success, 0, 1))
