library(data.table)
library(parallel)
###########################################################################
###   COMPILE_SUMMARY_TABLE.R                                           ###
###      * COMPILES SUMMARY STATISTICS FROM INDIVIDUAL MODEL RUNS       ###
###      * PRODUCES OUTPUT THAT IS FEED TO THE VISUALIZATION SHINY      ###
###      * HARDCODED TO EXPECTED DATA AND FILE LOCATIONS                ###
###                                                                     ###
###   USAGE:                                                            ###
###      Set run_name to match your Python run                          ###
###      Set run_type to "selection" or "validation"                    ###
###      Run this script separately for selection and validation runs   ###
###                                                                     ###
###      Contact: Amanda Meadows ~ amanda.meadows612@gmail.com          ###
###########################################################################

#### METADATA & OUTPUT LOCATIONS ###
# !UPDATE PATHS IF NECESSARY! 

# model configuration metadata file
meta <- fread("./model/input/metadata_example.csv")
# run name (should match the run_name used in fitOne.py)
run_name <- "test_run"
# run type: "selection" or "validation" - indicates which evaluation window this run corresponds to
run_type <- "selection"  # Change to "validation" for validation runs
# location of individual summary file output
output_loc <- paste0("./model/output/", run_name, "/")
# name for compiled output file (includes run_type for clarity)
outfile_name <- paste0("./model/output/", run_name, "_", run_type, "_compiled_summary.csv")

# Check if output directory exists
if (!dir.exists(output_loc)) {
  stop(paste("Error: Output directory not found:", output_loc))
}

scores_dir <- paste0(output_loc, "scores/")
if (!dir.exists(scores_dir)) {
  stop(paste("Error: Scores directory not found:", scores_dir))
}

###########################################

# list individual files
summ_files <- list.files(scores_dir, pattern = "_Summary.csv$", full.names = FALSE)

if (length(summ_files) == 0) {
  stop(paste("Error: No Summary files found in", scores_dir))
}

message(paste("Found", length(summ_files), "Summary files"))

# check to see if any rows did not write 
# may need to rerun the model if there was an error
completed_id <- as.numeric(gsub("_Summary.csv", "", summ_files))
missing_rows <- meta[!(ROW_ID %in% completed_id)]

if (nrow(missing_rows) > 0) {
  warning(paste("Warning: The following", nrow(missing_rows), "ROW_IDs from metadata do not have Summary files:"))
  print(missing_rows[, .(ROW_ID, country, model)])
}

read_summary_file <- function(filename){
  tryCatch({
    dt <- fread(paste0(scores_dir, filename), na.strings = "")
    dt[, file := filename]
    return(dt)
  },
  error = function(e) {
    # Handle error, e.g., file doesn't exist or cannot be read
    warning(paste("Could not read file", filename, ":", e$message))
    # Return NULL to be filtered out
    return(NULL)
  })
}

# Read all summary files in parallel
detectCores()
message("Reading summary files...")
summary_list <- mclapply(X = summ_files, FUN = read_summary_file, mc.cores = 10)

# Remove NULL results (failed reads)
summary_list <- summary_list[!sapply(summary_list, is.null)]

if (length(summary_list) == 0) {
  stop("Error: No summary files could be read successfully")
}

# Combine all summaries
test_dat <- rbindlist(summary_list, fill = TRUE)

# Add run_type flag to identify selection vs validation
test_dat[, run_type := run_type]
test_dat[, run_name := run_name]

# Write compiled output
fwrite(test_dat, outfile_name)
message(paste("Successfully compiled", nrow(test_dat), "summary records to", outfile_name))
message(paste("Run type:", run_type))