#!/usr/bin/env Rscript
###########################################################################
###   TEST_R_SYNTAX.R                                                    ###
###      * Tests R script syntax validation                             ###
###      * Validates that compilation scripts parse correctly           ###
###########################################################################

test_r_syntax <- function() {
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("R Script Syntax Tests\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  
  # Get base directory - handle different ways script might be called
  script_path <- commandArgs(trailingOnly = FALSE)
  if (any(grepl("--file=", script_path))) {
    script_file <- sub("--file=", "", script_path[grepl("--file=", script_path)][1])
    base_dir <- dirname(dirname(normalizePath(script_file)))
  } else {
    # Fallback: assume we're in test/ directory
    base_dir <- dirname(getwd())
  }
  
  # Test compilation scripts
  scripts_to_test <- c(
    "model_output_processing/1_compile_summary_table.R",
    "model_output_processing/2_compile_time_series_tables.R"
  )
  
  all_passed <- TRUE
  
  for (script in scripts_to_test) {
    script_path <- file.path(base_dir, script)
    if (file.exists(script_path)) {
      tryCatch({
        parse(script_path)
        cat("✓", basename(script), "syntax is valid\n")
      }, error = function(e) {
        cat("✗", basename(script), "has syntax error:", conditionMessage(e), "\n")
        all_passed <<- FALSE
      })
    } else {
      cat("⚠", basename(script), "not found (skipping)\n")
    }
  }
  
  cat(paste(rep("=", 60), collapse = ""), "\n")
  if (all_passed) {
    cat("✓ All R syntax tests passed\n")
    return(0)
  } else {
    cat("✗ Some R syntax tests failed\n")
    return(1)
  }
}

# Run tests
exit_code <- test_r_syntax()
quit(status = exit_code)
