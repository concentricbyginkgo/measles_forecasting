#### This script processes a variety of social predictor variables,
#### from various sources, for model fitting.
#### NOTE: see README_Social_Series.txt for a dictionary and additional
#### description of the data. 

library(data.table)
library(haven)

# 1) Update your local drive!
working_drive <- "~/measles_forecasting"
setwd(working_drive)

## NOTE: data in local_data/ are not included in repo (due to size constraints)
## and need to be downloaded manually. Update any paths to the location
## you stash these data files.
social_dat_drive <- "data_ingestion_pipeline/local_data/"

###########################
# id4d https://databank.worldbank.org/source/identification-for-development-(id4d)-data#
# pulled all countries, series, year available on 5/14/2023
id4d <- fread(paste0(social_dat_drive, "e318286f-5f90-48aa-9871-1c91a8b8a734_Data.csv"), na.strings = "")
setnames(id4d, c("Country Code", "Series Name", "Series Code"), c("ISO3", "Series_Name", "Series_Code"))
id4d <- id4d[!is.na(ISO3)]
id4d[, Series_Code := gsub("\\.", "_", Series_Code)]
id4d <- melt(id4d, id.vars = c("ISO3", "Country Name", "Series_Code", "Series_Name"))
id4d[, Year := as.numeric(substr(variable, 1, 4))]
id4d[, value_num := as.numeric(value)]

id4d <- id4d[, .(ISO3, Year, Series = Series_Code, value = value_num)]

###########################
# MAR Quantitative data http://www.mar.umd.edu/mar_data.asp (pulled on 5/14/2023)
# Current MAR Data
# The latest addition to the MAR dataset was released in February 2009 for the years 2004-2006.
# Codebook here: http://www.mar.umd.edu/data/mar_codebook_Feb09.pdf
# country can have more than 1 VMAR_Group

mar <- fread(paste0(social_dat_drive, "marupdate_20042006.csv"), na.strings = "")
mar[, ISO3 := countrycode::countrycode(sourcevar = country, origin = "country.name", destination = "iso3c")]
mar <- mar[!is.na(ISO3)] # removed Yugoslavia (no longer a country)
setnames(mar, "year", "Year")
setcolorder(mar, c("ISO3", "country", "Year"))

###########################
## State Capacity Scores https://statecapacityscores.org/the-data/ (national level pulled on 5/9/2023)
national_myers <- as.data.table(read_dta(social_dat_drive, "national_myers.dta"))
national_myers[, ISO3 := countrycode::countrycode(sourcevar = countryname, origin = "country.name", destination = "iso3c")]
setnames(national_myers, "year", "Year")
national_myers <- national_myers[, .(ISO3, Year, Series = "myers", value = myers)]


###########################
## Vaccine data
## AMEID (pulled on 5/14/2023)
# General Vaccination Policy and Childhood Vaccines https://ampeid.org/topics/general-vaccination-policies/, https://ampeid.org/topics/childhood-vaccination/
ameid_childhood <- fread(paste0(social_dat_drive, "AMP EID Childhood vaccination.csv"), na.strings = "")
ameid_general <- fread(paste0(social_dat_drive, "AMP EID General vaccination policies.csv"), na.strings = "")
ameid_childhood[, .N ,keyby = .(Subtopic, Status)]
ameid_general[, .N ,keyby = .(Subtopic, Status)]

ameid_childhood <- ameid_childhood[Subtopic == "Measles vaccination"]
ameid <- rbindlist(list(ameid_childhood, ameid_general), use.names = T)
setnames(ameid, "Status justification", "Status_justification")
ameid[, Subtopic := gsub(" ", "_", Subtopic)]
ameid <- ameid[, .(ISO3 = Country, Year = 2024, Series = Subtopic, value = Status)]

###########################
# UNICEF (pulled on 5/14/2023)
# https://data.unicef.org/topic/child-health/immunization/ Immunization by antigen
unicef_mcv1 <- as.data.table(readxl::read_excel(paste0(social_dat_drive, "wuenic2023rev_web-update.xlsx"), sheet = "MCV1"))
unicef_mcv2 <- as.data.table(readxl::read_excel(paste0(social_dat_drive, "wuenic2023rev_web-update.xlsx"), sheet = "MCV2"))

unicef <- rbindlist(list(unicef_mcv1, unicef_mcv2), use.names = T)

unicef_char <- unique(unicef[, .(ISO3 = iso3, Year = 2024, Series = "unicef_region", value = unicef_region)])
unicef[, c("unicef_region", "country") := NULL]
unicef <- melt(unicef, id.vars = c("iso3", "vaccine"))
unicef <- unicef[, .(ISO3 = iso3, Year = variable, Series = vaccine, value)]

###########################
## The world bank (pulled on 5/14/2023)
# https://www.who.int/data/gho/data/indicators/indicator-details/GHO/measles-containing-vaccine-first-dose-(mcv1)-immunization-coverage-among-1-year-olds-(-)
# https://www.who.int/data/gho/data/indicators/indicator-details/GHO/measles-containing-vaccine-second-dose-(mcv2)-immunization-coverage-by-the-nationally-recommended-age-(-)
wb_mcv1 <- fread(paste0(social_dat_drive, "mcv1_immunization_1yo.csv"), na.strings = "")
wb_mcv2 <- fread(paste0(social_dat_drive, "mcv2_immunization_recommended_age.csv"), na.strings = "")
wb_mcv1[, Series := "MCV1_Cov_1yo"]
wb_mcv2[, Series := "MCV2_Cov_RecAge"]

wb_mcv <- rbindlist(list(wb_mcv1, wb_mcv2), use.names = T)

wb_mcv <- wb_mcv[, .(ISO3 = SpatialDimValueCode, Year = Period, Series, value = Value)]


#################################################################################
# combine source data 
# Numerical data
social_predictors <- rbindlist(list(id4d, national_myers, unicef, wb_mcv), use.names = T)
social_predictors[, .N ,keyby = .(ISO3, Year, Series)][N != 1]
# Categorical data
social_predictors_categ <- rbindlist(list(ameid, unicef_char), use.names = T)
social_predictors_categ[, .N ,keyby = .(ISO3, Year, Series)][N != 1]
social_predictors_categ <- unique(social_predictors_categ)

# fwrite out datasets
fwrite(social_predictors, file = "./data_ingestion_pipeline/processed_data/social_predictors.csv", na = "")
fwrite(social_predictors_categ, file = "./data_ingestion_pipeline/processed_data/social_predictors_categ.csv", na = "")
fwrite(mar, file = "./data_ingestion_pipeline/processed_data/mar_predictors.csv", na = "")
