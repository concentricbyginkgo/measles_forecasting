# measles_forecasting
AI-enabled measles forecasting

## Data ingestion pipeline
The data ingestion pipeline contains scripts to process raw case and predictor data to a clean, consistent format for model fitting. The processed model data are provided, but the pipeline can be used to re-create, or refresh the provided processed dataset. Be advised that the processing of the gridded, temporal climate data is extremely compute and memory heavy. 

Manual downloading of the raw datasets is necessary, but links to the raw data are provided within the scripts. A data inventory with these links is also present in `other_deliverables/data_inventory/`. You can find additional information on the source of the data, temporal and spatial resolution of the data, license, etc. here as well.
