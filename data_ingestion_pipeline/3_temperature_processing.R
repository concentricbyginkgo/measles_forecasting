#### This script processes gridded daily maximum temperature data 
#### obtained from NOAA (see link below) for use in measles forecasting. 
#### The process involves geocoding gridded temperature data to a map 
#### and extracting the country code, which can be compute heavy and is best 
#### to run in parallel (see lines 91-93). You may need to tune the 
#### specifications to your machine.

# NOTE: This script will write and read several intermediary files 
# because the data are so large and the script is prone to crashing
# due to running out of compute power or memory. 

library(ncdf4)
library(ncdu)
library(data.table)
library(sp)
library(rgdal)
library(ggplot2)
library(rworldmap)
library(parallel)

# 1) Update your local drive!
working_drive <- "~/measles_forecasting/"
setwd(working_drive)

# 2) Download (manual step) 
## NOTE: data in local_data/ are not included in repo (due to size constraints)
## and need to be downloaded manually. Update any paths to the location
## you stash these data files. The data will be separated by year in the download
## due to the size of the data. It is recommended you upload the data to your local
## machine in data_ingestion_pipeline/local_data/

## Data download link: 
## https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html (choose maximum surface daily option)

# 3) Read in raw data and process
## Since temperature data is separated by year, a function was created
## so processing can iterate over the years of data. Several steps
## of processing are included in the function.

# extract_tmax: a function that reads raw downloaded temperature data
# and extracts the data to a data table.
## temp_path_in: the location within data_ingestion_pipeline/local_data where
## the temperature data are stored. e.g "data_ingestion_pipeline/local_data/t_max/"
## results_path_out: location to write yearly temperature data table. This
## is an intermediary step. The yearly data will be geocoded and
## summarized and combined to a single file later.
## year: the year of temperature data being extracted. 
## num_parallel_groups: the number of groups to divide the data for parallelization.
## The more groups, that smaller the memory footprint of each group.
## write_to_file: yes/no, should output be written file? 
extract_tmax <- function(temp_path_in, results_path_out, year, 
                         num_parallel_groups, write_to_file){
  
  print(year)
  
  pred_var <- "tmax"
  var_name <- "max_temp"
  filename <- paste0(temp_path_in, "tmax.", year, ".nc")
  
  nc_metadata <- ncdump::NetCDF(filename)
  t_origin <- gsub("hours since ", "", nc_metadata$unlimdims$units) 
  #t_origin <- '1900-01-01 00:00' # hours since 1900-01-01 00:00:00
  
  nc_data <- nc_open(filename)
  # Step 1: extract spatial and temporal dimensions of the data
  lon <- ncvar_get(nc_data, "lon")
  lon2 <- ifelse(lon > 180, lon - 360, lon)
  lat <- ncvar_get(nc_data, "lat", verbose = F)
  t <- ncvar_get(nc_data, "time")
  
  # figure out the origin for Julian dates stored in t by looking at metadata
  t_date <- as.Date(as.POSIXct(t*3600, origin = t_origin)) 
  
  # Step 2: extract the variable of interest
  var_array <- ncvar_get(nc_data, pred_var)
  fillvalue <- ncatt_get(nc_data, pred_var, "_FillValue")
  var_array[var_array == fillvalue$value] <- NA
  
  # Step 3: shape the arrays of data into a data table
  # used this guide: https://towardsdatascience.com/how-to-crack-open-netcdf-files-in-r-and-extract-data-as-time-series-24107b70dcd
  lonlattime <- as.matrix(expand.grid(lon,lat,t_date))
  var_vec_long <- as.vector(var_array)
  var_obs <- data.table(cbind(lonlattime, var_vec_long))
  var_obs[, Var1 := as.numeric(Var1)]
  var_obs[, Var2 := as.numeric(Var2)]
  var_obs[, Var3 := as.Date(Var3)]
  var_obs[, var_vec_long := as.numeric(var_vec_long)]
  # subset to years corresponding to measles case data
  var_dat <- var_obs[Var3 >= as.Date("2010-12-01")]
  var_dat <- var_dat[!is.na(var_vec_long)]
  var_dat[, Var1 := ifelse(Var1 >= 180, Var1 - 360, Var1)] # convert longitude from [0-360] to [-180-180]
  colnames(var_dat) <- c("lon", "lat", "date", var_name)
  # create parallel grouping id
  var_dat[, row := 1:.N]
  var_dat[, parallel_id := ceiling(row/(.N/num_parallel_groups))]
  var_dat[, .N, by = .(parallel_id)]
  # write to file and return data
  if(tolower(write_to_file) == "yes"){
    results_path_year <- paste0(results_path_out, "/", year, "/")
    fwrite(var_dat, paste0(results_path_year, "_", grp, ".csv"))
  }
  return(var_dat)
}


temp_table_dat <- extract_tmax(temp_path_in = "data_ingestion_pipeline/local_data/t_max/",
                               results_path_out = "data_ingestion_pipeline/processed_data/tmax/",
                               year = 2011, num_parallel_groups = 200, write_to_file = "yes")

# 5) Geocode temperature data

worldmap <- getMap(resolution='low') # map used for geocoding

# get_country: function to geocode temperature data to country
## dat: the temperature data table (created in step 5)
## grp: the parallel_id created above; cuts data table into smaller chunks
## out_location: the location where you want your geocoded data to be written

get_country <- function(dat, grp, out_location){
  dat_grp <- dat[parallel_id == grp, ]
  sp::coordinates(dat_grp) <- ~ lon2 + lat
  proj4string(dat_grp) <- proj4string(worldmap)
  world_over <- data.table(over(dat_grp, worldmap))
  dat_grp <- data.table(dat_grp@data, dat_grp@coords)
  out_dat <- cbind(dat_grp, world_over[, .(ISO2 = ISO_A2, ISO3 = ISO_A3, Name = ADMIN)])
  fwrite(out_dat, paste0(out_location, "tmax_", grp, ".csv"))
  return(out_dat)
}
# test a single group
get_country(dat = temp_table_dat, grp = 1,
            out_location = "data_ingestion_pipeline/processed_data/")

detectCores() # To decide how many cores to distribute job over
# X corresponds to grp; may need to update if you updated parallel_id in data table
t0 <- Sys.time()
precip_dat_list <- mclapply(get_country, dat = temp_table_dat, 
                            out_location = "data_ingestion_pipeline/processed_data/", 
                            X = 1:200, mc.cores = 4)
Sys.time()-t0


# 6) Read in geocoded data by group and compile to a single file for the year
finished_files <- list.files("data_ingestion_pipeline/processed_data/tmax/2011/")
read_temp <- function(filename){
  dat <- fread(paste0("data_ingestion_pipeline/processed_data/tmax/2011/", filename), na.strings = "")
return(dat)
}

geo_temp_list <- mclapply(read_temp, X = finished_files, mc.cores = 4)
geo_temp <- rbindlist(geo_temp_list)
geo_temp[, date := as.Date(date)]
fwrite(geo_temp, "data_ingestion_pipeline/processed_data/compiled_2011_tmax.csv")

# 7) Geocoding daily temperature data takes a long time,
# so recycle the first year of data and lookup lat/longs to get the country

##read in raw temperature data for remaining 
temp_year_table_list <- lapply(extract_tmax, "data_ingestion_pipeline/local_data/t_max/",
                               results_path_out = "data_ingestion_pipeline/processed_data/tmax/",
                               num_parallel_groups = 200,
                               write_to_file = "no",
                               X = 2012:2024)
temp_year_table <- rbindlist(temp_year_table)
latlonlookup <- unique(geo_temp[, .(ISO2, ISO3, Name, lat, lon)])
temp_year_table_list[latlonlookup, `:=`(ISO2 = i.ISO2, ISO3 = i.ISO3, Name = i.Name), on = .(lon, lat)]
outdat <- rbind(geo_temp, temp_year_table_list, use.names =T)
outdat[, date := as.Date(date)]
outdat[, month := lubridate::month(date)]
outdat[, year := lubridate::year(date)]
outdat[, date := as.Date(paste(month, "1", year, sep = "-"), format = "%m-%d-%Y")]

mean_tmax <- outdat[, .(mean_max_temp = mean(max_temp, na.rm = T)), by = .(ISO2, ISO3, Name, date)]
fwrite(mean_tmax, "data_ingestion_pipeline/processed_data/processed_tmax_data.csv")
