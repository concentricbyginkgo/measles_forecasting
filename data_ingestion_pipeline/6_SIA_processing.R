#### This script processes Supplementary Immunization Activity (SIA)
#### data from the WHO Global Immunization Database (GID)
#### for use in measles forecasting.

## !Download data from:! 
## How to Navigate to the dataset:
## Main Portal: Go to https://immunizationdata.who.int/
## All Data Section: Click on "All Data"
## Additional Datasets: Scroll down to find "Summary of Measles-Rubella Supplementary Immunization Activities"
## Click on "Download" to access the dataset

library(data.table)
library(readxl)
library(countrycode)

# 1) Update your local drive!
working_drive <- "~/measles_forecasting"
setwd(working_drive)
## NOTE: processed data in "local_data/" for individual predictors
## are not included in repo (due to size constraints)
## and need to be created by the scripts mentioned above. 
## Update any paths to the location you stash these data files.

local_dat_drive <- "data_ingestion_pipeline/local_data/"

# 2) Read from excel and clean data
SIA_dat <- fread(paste0(local_dat_drive, "V_SIA_MAIN_MR.csv"))
SIA_dat[, .N, by = .(INTERVENTION)]
SIA_dat[, .N, by = .(STATUS)]
#SIA_dat[STATUS == "Done" & is.na(END_DATE), ]
#SIA_dat[STATUS == "Planned", .(END_DATE)]
SIA_dat[, .N, by = .(ACTIVITY_TYPE)]
SIA_dat[, .N, by = .(EXTENT)]
SIA_dat[is.na(END_DATE)] # end date not always recorded
SIA_dat[is.na(START_DATE)]
SIA_dat[, start_date := lubridate::ymd(START_DATE)]
SIA_dat[, start_date := round_date(start_date, unit = "month")]
SIA_dat[is.na(start_date), start_date := lubridate::ymd(START_DATE, truncated = 2)]
SIA_dat[!is.na(start_date), .(START_DATE, start_date)]
SIA_dat[is.na(start_date), start_date := as.Date("2024-09-01")] # date didn't parse

SIA_dat[, end_date := lubridate::ymd(END_DATE)]
SIA_dat[, end_date := round_date(end_date, unit = "month")]
SIA_dat[, .(end_date, END_DATE)]
SIA_dat[is.na(end_date), end_date := lubridate::ymd(END_DATE, truncated = 2)]
SIA_dat[is.na(end_date), end_date := start_date]

SIA_summary <- unique(SIA_dat[, .(COUNTRY, YEAR, start_date, end_date)])
# some records show end date before start date, replace end with start date for these
SIA_summary[end_date < start_date, end_date := start_date]

# look for start months in Dec & roll forward to Jan of next year if
# end month doesn't also equal Dec
SIA_summary[, startm := month(start_date)]
SIA_summary[startm == 12 & start_date != end_date, start_date:= ceiling_date(start_date, "month")]
SIA_summary[startm == 12 & start_date != end_date, YEAR := YEAR + 1]

# get the earliest start month & the latest end month of the year
## a lot of entries are missing the end date
SIA_summary_start <- SIA_summary[, .SD[which.min(start_date)], by = .(COUNTRY, YEAR)]
SIA_summary_start[, end_date := NULL]
SIA_summary_end <- SIA_summary[, .SD[which.max(end_date)], by = .(COUNTRY, YEAR)]
# also get the latest start month (use if the latest end date is missing)
SIA_summary_start_max <- SIA_summary[, .SD[which.max(start_date)], by = .(COUNTRY, YEAR)]

# combine datasets to one row per country/year with earliest start and latest end date
SIA_summary_start[SIA_summary_end, end_date := i.end_date, on = .(COUNTRY, YEAR)]
SIA_summary_start[SIA_summary_start_max, max_start_date := i.start_date, on = .(COUNTRY, YEAR)]
SIA_summary_start[is.na(end_date), end_date := max_start_date]

# SIA status will be yes from the earliest start date to the latest end date within a calendar year
SIA_run <- SIA_summary_start[,.SD[CJ(start_date = seq(start_date, end_date, by = "month"), 
                                        unique = T),  on = .(start_date)], by = .(COUNTRY, YEAR)]
SIA_run[, STATUS := "yes"]

# 3) write to processed data folder
fwrite(SIA_run, "data_ingestion_pipeline/processed_data/SIA_summary.csv")
