#### This script processes measles case data (obtained from WHO; 
#### data link provided below) for model fitting. Key transformations 
#### include transforming data from wide to long format, and converting 
#### raw case counts to population-adjusted incidence.

library(data.table)
library(readxl)
library(tidyr)
library(lubridate)

# 1) Update your local drive!
working_drive <- "~/measles_forecasting"
setwd(working_drive)

# Create directory to write processed data
if(!dir.exists("data_ingestion_pipeline/processed_data/")){
  dir.create("data_ingestion_pipeline/processed_data/")
}

# 2) Download (manual step) and read in raw data
## NOTE: data in local_data/ are not included in repo (due to size constraints)
## and need to be downloaded manually. Update any paths to the location
## you stash these data files.

# Case data download link: https://immunizationdata.who.int/docs/librariesprovider21/measles-and-rubella/404-table-web-epi-curve-data.xlsx?sfvrsn=5922ebf7_4
## Note: the WHO data format changed sometime in early 2025, so shaping process has been updated to reflect this.
measlesDat <- data.table(read_xlsx("data_ingestion_pipeline/local_data/404-table-web-epi-curve-data.xlsx", sheet = "WEB", col_names = TRUE))

# Population data download link: https://population.un.org/wpp/Download/Standard/CSV/
PopDat <- fread("data_ingestion_pipeline/local_data/WPP2022_Demographic_Indicators_Medium.csv", na.strings = "") 
PopDat <- PopDat[!is.na(ISO2_code), ]

# 3) Subset data & format dates
measlesDat<- measlesDat[, .(ISO3, Country, Region, Year, Month, cases = `Measles \r\ntotal`)]
measlesDat[, mnth := as.numeric(Month)]
measlesDat[, Year := as.numeric(Year)]
measlesDat[, Month := lubridate::month(mnth, label = T, abbr = F)]
measlesDat[, date := lubridate::mdy(char_date)]
measlesDat[, cases := as.numeric(cases)]
measlesDatLong <- measlesDat[,.SD[CJ(date = seq(min(date), max(date), by = "month"), 
                                     unique = T),  on = .(date)], by = .(ISO3, Country, Region)]
setorder(measlesDatLong, "ISO3", "date")

# 4) Calculate cumulative cases and infer outbreak status
# function to set NA to 0 in cumulative sum, unless ALL values are NA
f1 <- function(x) if(all(is.na(x))) NA_real_ else cumsum(tidyr::replace_na(x, 0))
measlesDatLong[, cuml_cases := f1(cases), by = .(ISO3, Year)]

# add in UN demographic data 
measlesDatLong[PopDat, `:=`(population_jan = i.TPopulation1Jan*1000, 
                            population_july = i.TPopulation1July*1000,
                            birth_per_1k = i.CBR, births = i.Births*1000, 
                            migrations = i.NetMigrations*1000,
                            migrations_per_1k = i.CNMR), on = .(Year = Time, ISO3 = ISO3_code)]
measlesDatLong[, cases_1M := (cases/population_jan)*1000000]
measlesDatLong[, cuml_cases_1M := (cuml_cases/population_jan)*1000000]

# Determine if cases cross outbreak threshold
## CDC defines outbreak as >= 20 cases per million population https://stacks.cdc.gov/view/cdc/135224
measlesDatLong[, outbreak_20_per_M := ifelse(cases_1M >= 20, "yes", "no")]
measlesDatLong[is.na(outbreak_20_per_M), outbreak_20_per_M := "no"]
measlesDatLong[, outbreak_2_per_M := ifelse(cases_1M >= 2, "yes", "no")] 
measlesDatLong[is.na(outbreak_2_per_M), outbreak_2_per_M := "no"]
measlesDatLong[, outbreak_5_per_M := ifelse(cases_1M >= 5, "yes", "no")] 
measlesDatLong[is.na(outbreak_5_per_M), outbreak_5_per_M := "no"]
measlesDatLong[, outbreak_20_cuml_per_M := ifelse(cuml_cases_1M >= 20, "yes", "no")] 
measlesDatLong[is.na(outbreak_20_cuml_per_M), outbreak_20_cuml_per_M := "no"]

# 5) Determine months since last outbreak

get_months_since_last_outbreak <- function(var){
  tempDat <- copy(measlesDatLong)
  setnames(tempDat, var, "outbreak")
  
  # get the date of the first reported outbreak by country
  firstKnownOutbreak <- tempDat[outbreak == "yes", .SD[which.min(date)], by = .(ISO3)]
  tempDat[firstKnownOutbreak, date_first_outbreak := i.date, on = .(ISO3)]
  
  # Step 1: determine how many times the outbreak status of a country changes;
  ## rleid groups consecutive rows with the same value
  tempDat[, outbreak_grp := rleid(outbreak), by = .(ISO3)]
  # Step 2.0: count the number of months (each row = 1 country/month) since the outbreak 
  # status changed to get months since the last outbreak
  tempDat[, mnths_since_outbreak := 1:.N, by = .(ISO3, outbreak_grp)]
  ## Step 2.1: if outbreak status is yes, update to 0
  tempDat[, mnths_since_outbreak := ifelse(outbreak == "yes", 0, mnths_since_outbreak), by = .(ISO3)]
  ## Step 2.2: if a country has no outbreaks, set value to NA
  tempDat[, mnths_since_outbreak := ifelse(outbreak_grp == 1 & all(outbreak == "no"), NA_integer_, mnths_since_outbreak), by = .(ISO3)]
  ## Step 2.3: if the date is before the first known outbreak in dataset, set value to NA
  tempDat[, mnths_since_outbreak := ifelse(date < date_first_outbreak, NA_integer_, mnths_since_outbreak), by = .(ISO3)]
  ## Now, only the number of months since the last outbreak ended should have non-NA or non-zero values
  measlesDatLong[tempDat, paste0("mnths_since_", var) := i.mnths_since_outbreak, on = .(ISO3, date)]
  
  tempDat[tempDat[outbreak == "yes", uniqueN(outbreak_grp), by = .(ISO3)], num_outbreaks := i.V1, on =.(ISO3)]
  tempDat[is.na(num_outbreaks), num_outbreaks := 0]
  measlesDatLong[tempDat, paste0("num_", var) := i.num_outbreaks, on = .(ISO3)]
}

get_months_since_last_outbreak(var = "outbreak_20_per_M")
get_months_since_last_outbreak(var = "outbreak_2_per_M")
get_months_since_last_outbreak(var = "outbreak_20_cuml_per_M")

# 6 Proportion of months in outbreak 
measlesDatLong[, t := 1:.N, by = .(ISO3)]

### 5 cases per M
measlesDatLong[, outbreak_5_per_M := ifelse(cases_1M >= 5, 1, 0)]
measlesDatLong[is.na(outbreak_5_per_M)]
## Overall proportion of months in outbreak as of previous month
measlesDatLong[, cuml_mnths_outbreak_5M := cumsum(tidyr::replace_na(outbreak_5_per_M, 0)), by = .(ISO3)]
measlesDatLong[, prop_prev_mnths_in_outbreak_5M := shift(cuml_mnths_outbreak_5M, type = "lag", n = 1L)/shift(t, type = "lag", n = 1L), by = .(ISO3)]
## Proportion of months in outbreaks during 12 month rolling window as of previous month
measlesDatLong[, rolling_12_mnths_outbreak_5M := ifelse(t<12, cuml_mnths_outbreak_5M, frollsum(outbreak_5_per_M, n = 12)), by = .(ISO3)]
measlesDatLong[, prop_prev_rolling_12_mnths_outbreak_5M := shift(rolling_12_mnths_outbreak_5M, type = "lag", n = 1L)/12, by = .(ISO3)]
## Proportion of months in outbreaks during 24 month rolling window as of previous month
measlesDatLong[, rolling_24_mnths_outbreak_5M := ifelse(t<24, cuml_mnths_outbreak_5M, frollsum(outbreak_5_per_M, n = 24)), by = .(ISO3)]
measlesDatLong[, prop_prev_rolling_24_mnths_outbreak_5M := shift(rolling_24_mnths_outbreak_5M, type = "lag", n = 1L)/24, by = .(ISO3)]
## Proportion of months in outbreaks during 60 month rolling window as of previous month
measlesDatLong[, rolling_60_mnths_outbreak_5M := ifelse(t<60, cuml_mnths_outbreak_5M, frollsum(outbreak_5_per_M, n = 60)), by = .(ISO3)]
measlesDatLong[, prop_prev_rolling_60_mnths_outbreak_5M := shift(rolling_60_mnths_outbreak_5M, type = "lag", n = 1L)/60, by = .(ISO3)]

### 2 cases per M
measlesDatLong[, outbreak_2_per_M := ifelse(cases_1M >= 2, 1, 0)]
## Overall proportion of months in outbreak as of previous month
measlesDatLong[, cuml_mnths_outbreak_2M := cumsum(tidyr::replace_na(outbreak_2_per_M, 0)), by = .(ISO3)]
measlesDatLong[, prop_prev_mnths_in_outbreak_2M := shift(cuml_mnths_outbreak_2M, type = "lag", n = 1L)/shift(t, type = "lag", n = 1L), by = .(ISO3)]
## Proportion of months in outbreaks during 12 month rolling window as of previous month
measlesDatLong[, rolling_12_mnths_outbreak_2M := ifelse(t<12, cuml_mnths_outbreak_2M, frollsum(outbreak_2_per_M, n = 12)), by = .(ISO3)]
measlesDatLong[, prop_prev_rolling_12_mnths_outbreak_2M := shift(rolling_12_mnths_outbreak_2M, type = "lag", n = 1L)/12, by = .(ISO3)]
## Proportion of months in outbreaks during 24 month rolling window as of previous month
measlesDatLong[, rolling_24_mnths_outbreak_2M := ifelse(t<24, cuml_mnths_outbreak_2M, frollsum(outbreak_2_per_M, n = 24)), by = .(ISO3)]
measlesDatLong[, prop_prev_rolling_24_mnths_outbreak_2M := shift(rolling_24_mnths_outbreak_2M, type = "lag", n = 1L)/24, by = .(ISO3)]
## Proportion of months in outbreaks during 60 month rolling window as of previous month
measlesDatLong[, rolling_60_mnths_outbreak_2M := ifelse(t<60, cuml_mnths_outbreak_2M, frollsum(outbreak_2_per_M, n = 60)), by = .(ISO3)]
measlesDatLong[, prop_prev_rolling_60_mnths_outbreak_2M := shift(rolling_60_mnths_outbreak_2M, type = "lag", n = 1L)/60, by = .(ISO3)]

## Proportion of months in outbreaks during 60 month rolling window as of previous month
measlesDatLong[, rolling_12_mnths_mean_cases_1M := frollapply(cases_1M, 12, mean, na.rm = T), by = .(ISO3)]
measlesDatLong[, rolling_12_mnths_sd_cases_1M := frollapply(cases_1M, 12, sd, na.rm = T), by = .(ISO3)]
measlesDatLong[rolling_12_mnths_mean_cases_1M == 0][, .(ISO3, cases_1M)]

measlesDatLong[, rolling_36_mnths_mean_cases_1M := frollapply(cases_1M, 36, mean, na.rm = T), by = .(ISO3)]
measlesDatLong[, rolling_36_mnths_sd_cases_1M := frollapply(cases_1M, 36, sd, na.rm = T), by = .(ISO3)]
measlesDatLong[, rolling_60_mnths_mean_cases_1M := frollapply(cases_1M, 60, mean, na.rm = T), by = .(ISO3)]
measlesDatLong[, rolling_60_mnths_sd_cases_1M := frollapply(cases_1M, 60, sd, na.rm = T), by = .(ISO3)]
measlesDatLong[, cases_1M_12z := (cases_1M-shift(rolling_12_mnths_mean_cases_1M, n = 12, type = "lag"))/shift(rolling_12_mnths_sd_cases_1M, n = 12, type = "lag"), by = .(ISO3)]
measlesDatLong[, cases_1M_36z := (cases_1M-shift(rolling_36_mnths_mean_cases_1M, n = 12, type = "lag"))/shift(rolling_36_mnths_sd_cases_1M, n = 12, type = "lag"), by = .(ISO3)]
measlesDatLong[, cases_1M_60z := (cases_1M-shift(rolling_60_mnths_mean_cases_1M, n = 12, type = "lag"))/shift(rolling_60_mnths_sd_cases_1M, n = 12, type = "lag"), by = .(ISO3)]

# 7) write data 
measlesDatLong[, char_date := NULL]
fwrite(measlesDatLong, "data_ingestion_pipeline/processed_data/processed_measles_case_data.csv")


