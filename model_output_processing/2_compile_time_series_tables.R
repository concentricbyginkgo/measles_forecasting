library(data.table)
library(parallel)
library(lubridate)
###########################################################################
###   COMPILE_TIME_SERIE_DATA.R                                         ###
###      * COMPILES TIMER SERIES PROJECTION TABLES BY GEOGRAPHY         ###
###      * PRODUCES OUTPUT THAT IS FEED TO THE VISUALIZATION SHINY      ###
###      * HARDCODED TO EXPECTED DATA AND FILE LOCATIONS                ###
###                                                                     ###
###   USAGE:                                                            ###
###      Set run_name to match your Python run                          ###
###      Set run_type to "selection" or "validation"                   ###
###      This determines which subdirectory tables are written to       ###
###      Run this script separately for selection and validation runs   ###
###                                                                     ###
###      Contact: Amanda Meadows ~ amanda.meadows612@gmail.com          ###
###########################################################################

#### READ IN DATA ###
# !UPDATE PATHS IF NECESSARY! 

local_dir <- "~/python_projects/measles_forecasting/model/"
shiny_dir <- "~/python_projects/measles_forecasting/shiny_standalone/data/"

# Run name (should match the run_name used in fitOne.py)
run_name <- "test_run"
# run type: "selection" or "validation" - indicates which evaluation window this run corresponds to
# This determines which subdirectory the compiled tables will be written to
run_type <- "selection"  # Change to "validation" for validation runs

# Validate run_type
if (!run_type %in% c("selection", "validation")) {
  stop("Error: run_type must be either 'selection' or 'validation'")
}

# Create output directories if they don't exist
dir.create(paste0(shiny_dir, "tables/selection/"), recursive = TRUE, showWarnings = FALSE)
dir.create(paste0(shiny_dir, "tables/validation/"), recursive = TRUE, showWarnings = FALSE)
dir.create(paste0(shiny_dir, "validation_errors/"), recursive = TRUE, showWarnings = FALSE)

# observed case data
obs_file <- paste0(local_dir, "input/processed_measles_model_data.csv")
if (!file.exists(obs_file)) {
  stop(paste("Error: Observed data file not found:", obs_file))
}
obs_dat <- fread(obs_file)
if (!"date" %in% names(obs_dat) || !"ISO3" %in% names(obs_dat) || !"cases_1M" %in% names(obs_dat)) {
  stop("Error: Observed data file missing required columns: date, ISO3, cases_1M")
}
obs_dat[, date := as.Date(date)]

# train test split dates
tts_file <- paste0(local_dir, "input/final_model_validation_cutoffs.csv")
if (!file.exists(tts_file)) {
  warning(paste("Warning: TTS cutoff file not found:", tts_file, "- proceeding without cutoff dates"))
  tts_dat <- data.table(ISO3 = character(), cutoff_date = as.Date(character()), end_date = as.Date(character()))
} else {
  tts_dat <- fread(tts_file)
  if (!"cutoff_date" %in% names(tts_dat) || !"ISO3" %in% names(tts_dat)) {
    stop("Error: TTS data file missing required columns: cutoff_date, ISO3")
  }
  tts_dat[, cutoff_date := as.Date(cutoff_date)]
  tts_dat[, end_date := lubridate::add_with_rollback(cutoff_date, months(9))]
}

# Load compiled summary table (from script 1) or use metadata directly
# Try run_type-specific summary first, then generic summary, then metadata
summary_file <- paste0(local_dir, "output/", run_name, "_", run_type, "_compiled_summary.csv")
if (!file.exists(summary_file)) {
  # Try generic compiled summary (for backward compatibility)
  summary_file <- paste0(local_dir, "output/", run_name, "_compiled_summary.csv")
  if (!file.exists(summary_file)) {
    # Fallback to metadata if compiled summary doesn't exist
    warning(paste("Compiled summary file not found:", summary_file, "- using metadata file instead"))
    summary_file <- paste0(local_dir, "input/metadata_example.csv")
  }
}

if (!file.exists(summary_file)) {
  stop(paste("Error: Summary file not found:", summary_file))
}

summaryTable <- fread(summary_file)
if (!"ROW_ID" %in% names(summaryTable)) {
  stop("Error: Summary table missing required column: ROW_ID")
}

# Handle different column names for country ID
if ("ID" %in% names(summaryTable)) {
  country_col <- "ID"
} else if ("country" %in% names(summaryTable)) {
  country_col <- "country"
  summaryTable[, ID := country]
} else {
  stop("Error: Summary table missing country identifier column (ID or country)")
}

if ("file" %in% names(summaryTable)) {
  summaryTable[, file := NULL]
}

# Construct table filename based on new format: {ROW_ID}_{country}_Projection.csv
summaryTable[, table_filename := ifelse(!is.na(ROW_ID) & !is.na(ID), 
                                        paste0(ROW_ID, "_", ID, "_Projection.csv"), 
                                        NA_character_)]
summaryTable[, IDX := 1:.N]

####################################

read_tables <- function(summ_table, ROW, country){
  
  if (ROW > nrow(summ_table) || ROW < 1) {
    warning(paste("Invalid ROW index:", ROW))
    return(NULL)
  }
  
  table_file <- summ_table[IDX == ROW, ]$table_filename
  row_id <- ifelse("v_ROW_ID" %in% names(summ_table), summ_table[IDX == ROW, ]$v_ROW_ID, summ_table[IDX == ROW, ]$ROW_ID)
  iso3_id <- summ_table[IDX == ROW, ]$ID
  model_id <- ifelse("MODEL_ID" %in% names(summ_table), summ_table[IDX == ROW, ]$MODEL_ID, NA_character_)
  
  if(is.na(table_file) || table_file == ""){
    msg <- paste0("No table filename for ", iso3_id, ": ", model_id)
    warning(msg)
    return(NULL)
  }
  
  # Construct file path based on new format: output/{run_name}/tables/{ROW_ID}_{country}_Projection.csv
  # The table_file already contains ROW_ID and country, so we just need the run_name directory
  table_path <- paste0(local_dir, "output/", run_name, "/tables/", table_file)
  
  if (!file.exists(table_path)) {
    msg <- paste0("Table file not found for ", iso3_id, " (", model_id, "): ", table_path)
    warning(msg)
    error_dat <- data.table("ISO3" = iso3_id, "MODEL_ID" = model_id, "ROW_ID" = row_id, "error" = "file_not_found", "file_path" = table_path)
    error_filename <- paste0(shiny_dir, "validation_errors/", iso3_id, "_", model_id, ".csv")
    fwrite(error_dat, error_filename)
    return(NULL)
  }
  
  tryCatch({
    tables <- fread(table_path)
    
    # Validate required columns
    required_cols <- c("ds", "ID", "yhat1")
    missing_cols <- setdiff(required_cols, names(tables))
    if (length(missing_cols) > 0) {
      stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
    }
    
    tables[, ds := as.Date(ds)]
    
    # Merge with observed data
    if (nrow(obs_dat) > 0) {
      tables[obs_dat, obs_y := i.cases_1M, on = .(ID = ISO3, ds = date)]
    } else {
      tables[, obs_y := NA_real_]
    }
    
    # Calculate derived variables
    tables[, outbreak_observed_5M := ifelse(obs_y >= 5, "yes", "no")]
    tables[, outbreak_predicted_5M := ifelse(yhat1 >= 5, "yes", "no")]
    tables[, year := lubridate::year(ds)]
    
    # Calculate cumulative values by group
    if ("ROW_ID" %in% names(tables)) {
      tables[, cuml_y := cumsum(ifelse(is.na(obs_y), 0, obs_y)) + obs_y*0, by = .(ID, ROW_ID, year)]
      tables[, cuml_yhat1 := cumsum(ifelse(is.na(yhat1), 0, yhat1)) + yhat1*0, by = .(ID, ROW_ID, year)]
    } else {
      tables[, ROW_ID := row_id]
      tables[, cuml_y := cumsum(ifelse(is.na(obs_y), 0, obs_y)) + obs_y*0, by = .(ID, year)]
      tables[, cuml_yhat1 := cumsum(ifelse(is.na(yhat1), 0, yhat1)) + yhat1*0, by = .(ID, year)]
    }
    
    # Store run name and run type
    tables[, run := run_name]
    tables[, run_type := run_type]
    
    tables[, MODEL_ID := model_id]
    
    # Ensure 'y' column exists (Shiny app expects it) - use obs_y if y doesn't exist
    if (!"y" %in% names(tables) && "obs_y" %in% names(tables)) {
      tables[, y := obs_y]
    }
    
    # Select output columns (Shiny app expects: ROW_ID, MODEL_ID, ID, ds, obs_y, y, yhat1, etc.)
    output_cols <- c("ROW_ID", "MODEL_ID", "ID", "ds", "obs_y", "y", "yhat1", "year", "cuml_y", "cuml_yhat1", "outbreak_observed_5M", "outbreak_predicted_5M")
    available_cols <- intersect(output_cols, names(tables))
    tables_out <- tables[, ..available_cols]
    
    return(tables_out)
    
  }, error = function(e) {
    msg <- paste0("Error processing table for ", iso3_id, " (", model_id, "): ", e$message)
    warning(msg)
    error_dat <- data.table("ISO3" = iso3_id, 
                           "MODEL_ID" = model_id, 
                           "ROW_ID" = row_id,
                           "error" = e$message,
                           "file_path" = table_path)
    error_filename <- paste0(shiny_dir, "validation_errors/", iso3_id, "_", model_id, ".csv")
    fwrite(error_dat, error_filename)
    return(NULL)
  })
}

make_country_tables <- function(iso3,
                                evaluation_window = NULL, 
                                output_loc = NULL){
  
  # output_loc parameter kept for compatibility but not used (we write to subdirectories)
  
  tryCatch({
    # Get rows for this country
    country_rows <- summaryTable[ID == iso3]$IDX
    
    if (length(country_rows) == 0) {
      warning(paste("No rows found for country:", iso3))
      return(NULL)
    }
    
    # Read all tables for this country
    tables_list <- mclapply(country_rows, 
                           FUN = function(row) read_tables(summ_table = summaryTable, ROW = row, country = iso3),
                           mc.cores = 3)
    
    # Remove NULL results
    tables_list <- tables_list[!sapply(tables_list, is.null)]
    
    if (length(tables_list) == 0) {
      warning(paste("No valid tables found for country:", iso3))
      return(NULL)
    }
    
    # Combine all tables
    iso3_tables <- rbindlist(tables_list, fill = TRUE)
    
    if (nrow(iso3_tables) == 0) {
      warning(paste("Empty table for country:", iso3))
      return(NULL)
    }
    
    # Validate required columns
    if (!"ID" %in% names(iso3_tables) || !"ds" %in% names(iso3_tables)) {
      stop(paste("Missing required columns in compiled table for", iso3))
    }
    
    # Write country table to the appropriate subdirectory based on run_type
    output_filename <- paste0(shiny_dir, "tables/", run_type, "/", iso3, ".csv")
    
    fwrite(iso3_tables, output_filename)
    message(paste("Successfully wrote table for", iso3, "to", run_type, "directory"))
    
    # Merge with TTS cutoff dates if available
    if (nrow(tts_dat) > 0 && "ISO3" %in% names(tts_dat)) {
      iso3_tables[tts_dat, `:=`(cutoff_date = i.cutoff_date,
                                end_date = i.end_date), on = .(ID = ISO3)]
    }
    
    return(iso3_tables)
    
  }, error = function(e) {
    msg <- paste0("Table compilation for ", iso3, " failed: ", e$message)
    warning(msg)
    error_dat <- data.table("ISO3" = iso3, "error" = e$message, "timestamp" = Sys.time())
    error_filename <- paste0(shiny_dir, "validation_errors/", iso3, "_compilation_error.csv")
    fwrite(error_dat, error_filename)
    return(NULL)
  })
}

# Process all countries
countries <- sort(unique(summaryTable$ID))
if (length(countries) == 0) {
  stop("No countries found in summary table")
}

message(paste("Processing", length(countries), "countries..."))
results <- lapply(countries, FUN = make_country_tables)

# Summary
successful <- sum(!sapply(results, is.null))
message(paste("\nCompleted: Successfully processed", successful, "of", length(countries), "countries"))

