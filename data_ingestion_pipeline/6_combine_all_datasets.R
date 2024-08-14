#### This script combines all of the processed data created by
#### scripts 1_case_processing.R, 2_precip_processing.R,
#### 3_temperature_processing.R, 4_social_data_processing.R,
#### and 5_road_data_processing.R for use in measles forecast model training.

library(data.table)

# 1) Update your local drive!
working_drive <- "~/measles_forecasting"
setwd(working_drive)
## NOTE: processed data in "processed_data/" for individual predictors
## are not included in repo (due to size constraints)
## and need to be created by the scripts mentioned above. 
## Update any paths to the location you stash these data files.
processed_dat_drive <- "data_ingestion_pipeline/processed_data/"

MeaslesDat <- fread(paste0(processed_dat_drive, "processed_measles_case_data.csv"), na.strings = "")
MeaslesDat[, date := as.Date(date)]
MeaslesDat <- MeaslesDat[date < as.Date("2024-04-01")] # Data are mostly NAs beyond this point, change for data refresh

TmaxDat <- fread(paste0(processed_dat_drive, "processed_tmax_data.csv"), na.strings = "")
TmaxDat[, date := as.Date(date)]
MeaslesDat[TmaxDat, mean_max_temp := i.mean_max_temp, on = .(date, ISO3)]
MeaslesDat[is.na(mean_max_temp), .N, by = .(Country)]

PrecipDat <- fread(paste0(processed_dat_drive, "processed_precip_data.csv"), na.strings = "")
PrecipDat[, date := as.Date(date)]
MeaslesDat[PrecipDat, `:=`(mean_precip_mm_per_day = i.mean_precip_mm_per_day,
                           total_precip_mm_per_day = i.total_precip_mm_per_day), on = .(date, ISO3)]
MeaslesDat[is.na(mean_precip_mm_per_day), .N, by = .(Country)]

NumSocialDat <- fread(paste0(processed_dat_drive, "social_predictors.csv"), na.strings = "")
NumSocialDat_wide <- dcast(NumSocialDat, ISO3 + Year ~ Series, value.var = "value")
MeaslesDat <- merge(MeaslesDat, NumSocialDat_wide, by = c("ISO3", "Year"), all.x = T)

CatSocialDat <- fread(paste0(processed_dat_drive, "social_predictors_categ.csv"), na.strings = "")
enforcement_CatSocialDat <- CatSocialDat[Series %in% c("Enforcement_of_childhood_vaccination_requirement",
                                                       "Enforcement_of_emergency_vaccination")]
enforcement_CatSocialDat[, Num_mechanism := 1:.N, by = .(ISO3, Series)]
enforcement_CatSocialDat[, Series := paste0(Series, "_", Num_mechanism)]
enforcement_CatSocialDat_wide <- dcast(enforcement_CatSocialDat, ISO3 + Year ~ Series, value.var = "value", )
enforcement_CatSocialDat_wide[, Year := NULL]

CatSocialDat <- CatSocialDat[!(Series %in% c("Enforcement_of_childhood_vaccination_requirement",
                                             "Enforcement_of_emergency_vaccination"))]
CatSocialDat_wide <- dcast(CatSocialDat, ISO3 + Year ~ Series, value.var = "value", )

CatSocialDat_wide[, Year := NULL] # will not merge on year since it is a static, categorical variable
MeaslesDat <- merge(MeaslesDat, CatSocialDat_wide, by = c("ISO3"), all.x = T)
MeaslesDat <- merge(MeaslesDat, enforcement_CatSocialDat_wide, by = c("ISO3"), all.x = T)

roadsDat <- fread(paste0(processed_dat_drive, "groads_road_length.csv"), na.strings = "")
MeaslesDat[roadsDat, total_road_length_km := i.total_road_length_km, on = .(ISO3)]

fwrite(MeaslesDat, "model_training_data.csv")
