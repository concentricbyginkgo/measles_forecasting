#### This script processes gridded monthly precipitation data 
#### obtained from NOAA (see link below) for use in measles forecasting. 
#### The process involves geocoding gridded precipitation data to a map 
#### and extracting the country code, which can be compute heavy and is best 
#### to run in parallel (see Step 6). You may need to tune the 
#### specifications to your machine.

library(ncdf4)
library(ncdu)
library(data.table)
library(sp)
library(rgdal)
library(rworldmap)
library(parallel)

# 1) Update your local drive!
working_drive <- "~/measles_forecasting/"
setwd(working_drive)

# Create directory to write processed data
if(!dir.exists("data_ingestion_pipeline/processed_data/")){
  dir.create("data_ingestion_pipeline/processed_data/")
}

# 2) Download (manual step) and read in raw data
## NOTE: data in local_data/ are not included in repo (due to size constraints)
## and need to be downloaded manually. Update any paths to the location
## you stash these data files.

# Data download link: https://psl.noaa.gov/data/gridded/data.cmap.html (choose monthly mean option)
nc_data <- nc_open("data_ingestion_pipeline/local_data/precip.mon.mean.nc") # update data location to your 

# 3) extract spatial and temporal dimensions of the data
lon <- ncvar_get(nc_data, "lon")
lat <- ncvar_get(nc_data, "lat", verbose = F)
t <- ncvar_get(nc_data, "time")

# figure out the origin for Julian dates stored in t by looking at metadata
nc_metadata <- ncdump::NetCDF("data_ingestion_pipeline/local_data/precip.mon.mean.nc")
nc_metadata # hours since 1800-01-01 00:00
t_date <- as.Date(as.POSIXct(t*3600,origin='1800-01-01 00:00'))

# 4) extract the precip data
precip_array <- ncvar_get(nc_data, "precip")
fillvalue <- ncatt_get(nc_data, "precip", "_FillValue")
precip_array[precip_array == fillvalue$value] <- NA

# 5) Shape the arrays of data into a data table
## I used this guide: 
## https://towardsdatascience.com/how-to-crack-open-netcdf-files-in-r-and-extract-data-as-time-series-24107b70dcd
lonlattime <- as.matrix(expand.grid(lon,lat,t_date))
precip_vec_long <- as.vector(precip_array)
precip_obs <- data.table(cbind(lonlattime, precip_vec_long))
colnames(precip_obs) <- c("lon", "lat", "date", "precip_mm_per_day")
precip_obs[, lon := as.numeric(lon)]
precip_obs[, lat := as.numeric(lat)]
precip_obs[, precip_mm_per_day := as.numeric(precip_mm_per_day)]
precip_obs[, date := as.Date(date)]
precip_dat <- precip_obs[date >= as.Date("2010-12-01")] # only need data since 2011 to match available case data
precip_dat <- precip_dat[!is.na(precip_mm_per_day)]
precip_dat[, lon2 := ifelse(lon >= 180, lon - 360, lon)] # convert from [0-360] to [-180-180]

# 6) geocode spatial coordinates to map and match to country

## Geocoding can take a long time, so divide up the data table
## into smaller chunks that can be parallelized. The number 
## might need tuning to the specs of your machine for 
## optimal performance.
precip_dat[, row := 1:.N]
precip_dat[, parallel_id := ceiling(row/(.N/10))] 

# map used for geocoding
worldmap <- getMap(resolution='low')

# get_country: function to geocode precipitation data to country
## dat: the precipitation data table (created in step 5)
## grp: the parallel_id created above; cuts data table into smaller chunks
## out_location: the location where you want your geocoded data to be written

get_country <- function(dat, grp, out_location){
  dat_grp <- dat[parallel_id == grp, ]
  sp::coordinates(dat_grp) <- ~ lon2 + lat
  proj4string(dat_grp) <- proj4string(worldmap)
  world_over <- data.table(over(dat_grp, worldmap))
  dat_grp <- data.table(dat_grp@data, dat_grp@coords)
  out_dat <- cbind(dat_grp, world_over[, .(ISO2 = ISO_A2, ISO3 = ISO_A3, Name = ADMIN)])
  fwrite(out_dat, paste0(out_location, "precip_", grp, ".csv"))
  return(out_dat)
}

# #test a single group
# get_country(dat = precip_dat, grp = 1,
#             out_location = "data_ingestion_pipeline/processed_data/")

detectCores() # To decide how many cores to distribute job over
# X corresponds to grp; may need to update if you updated parallel_id in data table
t0 <- Sys.time()
precip_dat_list <- mclapply(get_country, dat = precip_dat, 
                            out_location = "data_ingestion_pipeline/processed_data/", 
                            X = 1:10, mc.cores = 2)
Sys.time()-t0

# compile_data: a function to read in each data file written
# by get_country above, and compile to a single data table
compile_data <- function(grp, out_location){
  dat <- fread(paste0(out_location, "precip_", grp, ".csv"))
  return(dat)
}

precip_dat_iso2 <- rbindlist(lapply(compile_data, X = 1:10))
precip_dat_iso2[Name == "Namibia", ISO2 := "NA"]

# If you want to write the data before taking the mean value by country, do it now
#fwrite(precip_dat_iso2, "data_ingestion_pipeline/processed_data/compiled_precip_iso2.csv")

precip_summ <- precip_dat_iso2[, .(mean_precip_mm_per_day = mean(precip_mm_per_day),
                    total_precip_mm_per_day = sum(precip_mm_per_day)), by = .(date, ISO2, ISO3, Name)]
fwrite(precip_summ, "data_ingestion_pipeline/processed_data/processed_precip_data.csv")

