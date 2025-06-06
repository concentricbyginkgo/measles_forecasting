#### This script combines all of the processed data created by
#### scripts 1_case_processing.R, 2_precip_processing.R,
#### 3_temperature_processing.R, 4_social_data_processing.R,
#### and 5_road_data_processing.R for use in measles forecast model training.

library(data.table)

# =============================================================================
# SETUP: Working directory and data paths
# =============================================================================

# 1) Update your local drive!
working_drive <- "~/measles_forecasting"
setwd(working_drive)
## NOTE: processed data in "processed_data/" for individual predictors
## are not included in repo (due to size constraints)
## and need to be created by the scripts mentioned above. 
## Update any paths to the location you stash these data files.
processed_dat_drive <- "data_ingestion_pipeline/processed_data/"

# =============================================================================
# LOAD BASE MEASLES CASE DATA 
# =============================================================================

# Load the main measles case dataset (long format from script 1)
# This serves as the backbone dataset - all other data will be merged into this
MeaslesDat <- fread(paste0(processed_dat_drive, "processed_measles_case_data.csv"), na.strings = "")
MeaslesDat[, date := as.Date(date)]
# Filter to data before April 2024 due to data quality issues beyond this point
MeaslesDat <- MeaslesDat[date < as.Date("2024-04-01")] # Data are mostly NAs beyond this point, change for data refresh

# =============================================================================
# MERGE CLIMATE DATA: Temperature and Precipitation
# =============================================================================

# Load and merge temperature data
# Temperature data is matched by date and country (ISO3 code)
TmaxDat <- fread(paste0(processed_dat_drive, "processed_tmax_data.csv"), na.strings = "")
TmaxDat[, date := as.Date(date)]
# Left join: add temperature data to measles dataset where dates/countries match
MeaslesDat[TmaxDat, mean_max_temp := i.mean_max_temp, on = .(date, ISO3)]
# Check how many countries have missing temperature data (for quality assessment)
MeaslesDat[is.na(mean_max_temp), .N, by = .(Country)]

# Load and merge precipitation data  
# Similar approach: match by date and country
PrecipDat <- fread(paste0(processed_dat_drive, "processed_precip_data.csv"), na.strings = "")
PrecipDat[, date := as.Date(date)]
# Add both mean and total precipitation metrics in one operation
MeaslesDat[PrecipDat, `:=`(mean_precip_mm_per_day = i.mean_precip_mm_per_day,
                           total_precip_mm_per_day = i.total_precip_mm_per_day), on = .(date, ISO3)]
# Check coverage of precipitation data
MeaslesDat[is.na(mean_precip_mm_per_day), .N, by = .(Country)]

# =============================================================================
# MERGE SOCIAL/ECONOMIC DATA: Numeric and Categorical Variables
# =============================================================================

# Process numeric social indicators (GDP, population density, etc.)
# Data comes in long format, needs to be pivoted to wide format for merging
NumSocialDat <- fread(paste0(processed_dat_drive, "social_predictors.csv"), na.strings = "")
# Pivot from long to wide: each 'Series' becomes a column, values spread across ISO3+Year
NumSocialDat_wide <- dcast(NumSocialDat, ISO3 + Year ~ Series, value.var = "value")
# Merge by country and year (left join to preserve all measles data)
MeaslesDat <- merge(MeaslesDat, NumSocialDat_wide, by = c("ISO3", "Year"), all.x = T)

# Process categorical social indicators (policy variables, etc.)
# More complex due to enforcement variables that can have multiple entries per country
CatSocialDat <- fread(paste0(processed_dat_drive, "social_predictors_categ.csv"), na.strings = "")

# Special handling for enforcement variables (can have multiple mechanisms per country)
# Extract enforcement-related variables for separate processing
enforcement_CatSocialDat <- CatSocialDat[Series %in% c("Enforcement_of_childhood_vaccination_requirement",
                                                       "Enforcement_of_emergency_vaccination")]
# Add sequence numbers to handle multiple enforcement mechanisms per country
enforcement_CatSocialDat[, Num_mechanism := 1:.N, by = .(ISO3, Series)]
# Create unique column names for each enforcement mechanism
enforcement_CatSocialDat[, Series := paste0(Series, "_", Num_mechanism)]
# Pivot to wide format
enforcement_CatSocialDat_wide <- dcast(enforcement_CatSocialDat, ISO3 + Year ~ Series, value.var = "value", )
# Remove Year column since enforcement data is country-level (not time-varying)
enforcement_CatSocialDat_wide[, Year := NULL]

# Process remaining categorical variables (non-enforcement)
# Remove enforcement variables from main categorical dataset to avoid duplication
CatSocialDat <- CatSocialDat[!(Series %in% c("Enforcement_of_childhood_vaccination_requirement",
                                             "Enforcement_of_emergency_vaccination"))]
# Pivot to wide format
CatSocialDat_wide <- dcast(CatSocialDat, ISO3 + Year ~ Series, value.var = "value", )

# Remove Year from categorical data since these are mostly time-invariant country characteristics
CatSocialDat_wide[, Year := NULL] # will not merge on year since it is a static, categorical variable
# Merge both categorical datasets (merge by ISO3 only, not year)
MeaslesDat <- merge(MeaslesDat, CatSocialDat_wide, by = c("ISO3"), all.x = T)
MeaslesDat <- merge(MeaslesDat, enforcement_CatSocialDat_wide, by = c("ISO3"), all.x = T)

# =============================================================================
# MERGE INFRASTRUCTURE DATA: Road Network Data
# =============================================================================

# Load and merge road infrastructure data
# This provides a proxy for connectivity/development level by country
roadsDat <- fread(paste0(processed_dat_drive, "groads_road_length.csv"), na.strings = "")
# Add road length data by country (time-invariant)
MeaslesDat[roadsDat, total_road_length_km := i.total_road_length_km, on = .(ISO3)]

# =============================================================================
# MERGE VACCINATION CAMPAIGN DATA: Supplementary Immunization Activities (SIA)
# =============================================================================

# Load WHO SIA (vaccination campaign) data
# This captures mass vaccination campaigns that can impact disease dynamics
SIA_dat <- fread(paste0(processed_dat_drive, "SIA_summary.csv"), na.strings = "")

# Merge SIA status with measles data by country, year, and campaign start date
# Note: using SIA_run variable (seems to be a processed version of SIA_dat)
MeaslesDat[SIA_run, SIA_status := i.STATUS, on = .(ISO3 = COUNTRY, Year = YEAR, date = start_date)]
# Set countries/periods with no SIA activity to "no" 
MeaslesDat[is.na(SIA_status), SIA_status := "no"]

# Calculate months since last SIA campaign (similar logic to outbreak calculations)
# Step 1: Create grouping variable for consecutive months with same SIA status
MeaslesDat[, sia_grp := rleid(SIA_status), by = .(ISO3)]
# Step 2: Count months within each SIA status group
MeaslesDat[, mnths_since_SIA := 1:.N, by = .(ISO3, sia_grp)]
# Step 3: If currently in SIA period, set months since SIA to 0
MeaslesDat[, mnths_since_SIA := ifelse(SIA_status == "yes", 0, mnths_since_SIA), by = .(ISO3)]
# Step 4: For countries that never had SIA campaigns, set to NA
MeaslesDat[, mnths_since_SIA := ifelse(sia_grp == 1 & all(SIA_status == "no"), NA_integer_, mnths_since_SIA), by = .(ISO3)]

# =============================================================================
# MERGE CLUSTER DATA: Country cluster assignments
# =============================================================================

clusterDat <- fread("data_ingestion_pipeline/provided_data/clusterDat.csv", na.strings = "")
MeaslesDat[clusterDat, cluster := i.cluster, on = .(ISO3)]
MeaslesDat[clusterDat, cluster_region := i.cluster_region, on = .(ISO3)]
MeaslesDat[clusterDat, cluster_redrawn := i.cluster_redrawn, on = .(ISO3)]

# =============================================================================
# EXPORT FINAL COMBINED DATASET
# =============================================================================

# Write the final combined dataset for model training
# This file contains all predictors aligned with measles case data
fwrite(MeaslesDat, "model_training_data.csv")
