# Data ingestion pipeline

The scripts here will recreate `model_training_data.csv`. Scripts 1-5 must be successfully run before running `6_combine_all_datasets.R`. Script `4_social_data_processing.R` has it's own README file `README_Social_Series.txt` because there are several datasets processed within that script. 

Note that `2_precip_processing.R` and `3_temperature_processing.R` extract and geocode gridded, temporal data (from `.nc` files to data table). This process is compute and memory intensive. Precipitation data is available on a monthly temporal resolution, while temperature data is at a daily resolution. While both datasets take considerable compute time to process, the temperature dataset is especially intensive. The scripts are set up to run parallelized, so be sure to tune these parameters to the specifications of your machine.
