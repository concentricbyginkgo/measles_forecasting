### This script processes monthly average temperature data obtained from Our World in Data ###
# for use in measles forecasting. 
# Data can be viewed at https://ourworldindata.org/grapher/monthly-average-surface-temperatures-by-year?country=~AFG
# but is accessible via api.
# Contains modified Copernicus Climate Change Service information (2025) – with major processing by Our World in Data

library(data.table)
## Fetch the metadata
#metadata <- fromJSON("https://ourworldindata.org/grapher/monthly-average-surface-temperatures-by-year.metadata.json?v=1&csvType=full&useColumnShortNames=false")
#metadata$columns

tempDat <- fread("https://ourworldindata.org/grapher/monthly-average-surface-temperatures-by-year.csv?v=1&csvType=full&useColumnShortNames=false")
tempDatLong <- melt(tempDat, id.vars = c("Entity", "Code", "Year"))
names(tempDatLong) <- c("Country", "ISO3", "Month", "Year", "mean_temp")
tempDatLong[, `:=`(Month = as.numeric(Month),
                   Year = as.numeric(as.character(Year)))]
tempDatLong <- tempDatLong[Year >= 2011, ]
tempDatLong[, date := lubridate::mdy(paste0(Month, "/1/", Year))]
tempDatLong <- tempDatLong[!is.na(mean_temp)]
fwrite(mean_tmax, "data_ingestion_pipeline/processed_data/processed_temp_data.csv")
