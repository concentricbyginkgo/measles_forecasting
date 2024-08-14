#### This script processes road length data from SEDAC
#### for use in measles forecasting.

## !Download data from:! 
## https://sedac.ciesin.columbia.edu/downloads/docs/groads/groads-v1-source-and-road-length-by-country.xlsx

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
groad <- as.data.table(read_excel(paste0(local_dat_drive, "groads-v1-source-and-road-length-by-country.xlsx"), sheet = "Source & Length", range = c("A7:B221")))
groad <- groad[!is.na(Country)]
groad[, ISO3 := countrycode(sourcevar = Country, origin = "country.name", destination = "iso3c")]
groad[Country == "Micronesia", ISO3 := "FSM"]
setnames(groad, "Total Road Length(km)", "total_road_length_km")

# 3) write to processed data folder
fwrite(groad, "data_ingestion_pipeline/processed_data/groads_road_length.csv")
