#### This script processes measles case data (obtained from WHO; 
#### data link provided below) for model fitting. Key transformations 
#### include transforming data from wide to long format, and converting 
#### raw case counts to population-adjusted incidence.

library(data.table)
library(readxl)

# 1) Update your local drive!
working_drive <- "~/measles_forecasting"
setwd(working_drive)

# 2) Download (manual step) and read in raw data
## NOTE: data in local_data/ are not included in repo (due to size constraints)
## and need to be downloaded manually. Update any paths to the location
## you stash these data files.

# Case data download link: http://immunizationdata.who.int/docs/librariesprovider21/measles-and-rubella/407-table-web-measles-cases-by-month.xlsx?sfvrsn=41bda8f6_1
measlesDat <- data.table(read_xlsx("data_ingestion_pipeline/local_data/measlescountrymnth.xlsx", sheet = "WEB"))

# Population data download link: https://population.un.org/wpp/Download/Standard/CSV/
PopDat <- fread("data_ingestion_pipeline/local_data/WPP2022_Demographic_Indicators_Medium.csv", na.strings = "") 
PopDat <- PopDat[!is.na(ISO2_code), ]

# 3) Shape data to long format & format dates
measlesDatLong <- melt(measlesDat, id.vars = c("Region", "ISO3", "Country", "Year"), variable.name = "Month", value.name = "cases")
measlesDatLong[, mnth := as.numeric(match(Month, month.name))]
measlesDatLong[, char_date := paste0(mnth, "/1/", Year)]
measlesDatLong[, date := lubridate::mdy(char_date)]
measlesDatLong[, cases := as.numeric(cases)]
measlesDatLong[, month_factor := factor(Month, levels = month.name)]
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

# 6) write data 
# Determine number of outbreaks (since 2011) in each country
measlesDatLong[, `:=`(month_factor = NULL, char_date = NULL)]
fwrite(measlesDatLong, "data_ingestion_pipeline/processed_data/processed_measles_case_data.csv")
