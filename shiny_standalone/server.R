############################################
# MEASLES MODEL VALIDATION SHINY APP - SERVER SCRIPT
# 
# This script contains the server-side logic for the measles model validation Shiny app.
# It handles:
# - Reactive data processing and filtering
# - Model performance table generation
# - Interactive plot rendering (epidemiological curves and binary outcomes)
# - Data visualization for both model selection and validation periods
#
# Author: [Author Name]
# Date: [Date]
# Version: 1.0
############################################

# Define server logic for the Shiny application
shinyServer(function(input, output, session) {
  
  # Render epidemiological curves for model selection period
  # Shows observed vs predicted case counts with model performance comparison
  output$epi_curves <- renderPlotly({
    
    COUNTRY <- input$COUNTRY
    if(is.na(input$top_n))
      return
    
    # Get top N models based on user selection
    n_to_take <- min(input$top_n, nrow(summaryTable[ID == COUNTRY,]))
    top_iso3_plot_dat <- get_plot_dat(summ_dt = summaryTable, cutoff_dat = cutoff_dat,
                                      iso3 = COUNTRY, col_name = input$order_summary_col_by, n = n_to_take, by_config = "yes")
    
    # Sort model IDs and convert to factor for consistent ordering
    model_ids <- sort(unique(top_iso3_plot_dat$MODEL_ID))
    top_iso3_plot_dat[, MODEL_ID := factor(MODEL_ID, levels = model_ids)]
    
    # Convert dates and extract numeric values for vertical lines
    top_iso3_plot_dat[, `:=`(cutoff_date = as.Date(cutoff_date), end_date = as.Date(end_date))]
    cutoff_num <- as.numeric(top_iso3_plot_dat[run_period == "selection"]$cutoff_date[1])
    end_num    <- as.numeric(top_iso3_plot_dat[run_period == "selection"]$end_date[1])
    
    # Create epidemiological curves plot for selection period
    epi_curves <- ggplot(top_iso3_plot_dat[run_period == "selection"]) +
      # Observed data line
      geom_line(aes(y = y, x = ds, color = "Observed")) +
      # Model prediction lines
      geom_line(aes(y = yhat1, x = ds, group = MODEL_ID, color = MODEL_ID, linetype = is_cluster_run), alpha = 0.7) +
      # Vertical lines for cutoff and end dates
      geom_vline(xintercept = cutoff_num, color = "firebrick") +
      geom_vline(xintercept = end_num, color = "firebrick") +
      # Color scale for models
      scale_color_manual(
        values = c(
          "Observed" = "black",
          setNames(
            viridis::viridis(min(n_to_take, length(model_ids))),
            as.character(model_ids[seq_len(min(n_to_take, length(model_ids)))])
          )
        ),
        breaks = c("Observed", as.character(model_ids)),
        labels = c("Observed", paste0("MODEL_ID:", model_ids))) +
      # Line type scale (solid for individual runs, dashed for cluster runs)
      scale_linetype_manual(
        values = c("no" = "solid", "yes" = "dashed"),
        guide = "none"
      ) +
      # Labels and title
      labs(
        title = top_iso3_plot_dat$ID[1],
        subtitle = paste(top_iso3_plot_dat$model[1], "\n", top_iso3_plot_dat$predictor[1]),
        x = "Date", y = "Cases per 1M", color = "Series"
      ) +
      theme_minimal() +
      theme(
        legend.position = "bottom",
        plot.title = element_text(size = 12, hjust = 0.5),
        plot.subtitle = element_text(size = 8, hjust = 0.5)
      )
    
    return(epi_curves)
    
  })
  
  # Render binary outcome heatmap for model selection period
  # Shows predicted vs observed outbreak classifications (5M threshold)
  output$binary_outcome_plots <- renderPlotly({
    
    COUNTRY <- input$COUNTRY
    if(is.na(input$top_n))
      return
    
    # Get top N models based on user selection
    n_to_take <- min(input$top_n, nrow(summaryTable[ID == COUNTRY,]))
    top_iso3_plot_dat <- get_plot_dat(summ_dt = summaryTable, cutoff_dat = cutoff_dat,
                                      iso3 = COUNTRY, col_name = input$order_summary_col_by, n = n_to_take, by_config = "yes")
    
    # Generate binary outcome heatmap for selection period
    binary_outcome_analysis <- plot_binary_outcome(plot_dat = top_iso3_plot_dat, period = "selection")
    
    return(binary_outcome_analysis)
    
  })
  
  # Render model evaluation data table
  # Shows performance metrics for top N models ordered by selected criteria
  output$model_evaluation_table <- renderDataTable({
    n_to_take <- min(input$top_n, nrow(examineDat[ID == input$COUNTRY,]))
    subset_country <- examineDat[ID == input$COUNTRY]
    setorderv(subset_country, cols = input$order_summary_col_by)
    subset_country <- subset_country[, head(.SD, n_to_take)]
    return(subset_country)
  })
  
  # Render epidemiological curves for model validation period
  # Shows observed vs predicted case counts for validation data
  output$epi_curves_validation <- renderPlotly({
    
    COUNTRY <- input$COUNTRY
    if(is.na(input$top_n))
      return
    
    # Get top N models based on user selection
    n_to_take <- min(input$top_n, nrow(summaryTable[ID == COUNTRY,]))
    top_iso3_plot_dat <- get_plot_dat(summ_dt = summaryTable, cutoff_dat = cutoff_dat,
                                      iso3 = COUNTRY, col_name = input$order_summary_col_by, n = n_to_take, by_config = "yes")
    
    # Sort model IDs and convert to factor for consistent ordering
    model_ids <- sort(unique(top_iso3_plot_dat$MODEL_ID))
    top_iso3_plot_dat[, MODEL_ID := factor(MODEL_ID, levels = model_ids)]
    
    # Convert dates and extract numeric values for vertical lines
    top_iso3_plot_dat[, `:=`(cutoff_date = as.Date(cutoff_date), end_date = as.Date(end_date))]
    cutoff_num <- as.numeric(top_iso3_plot_dat[run_period == "validation"]$cutoff_date[1])
    end_num    <- as.numeric(top_iso3_plot_dat[run_period == "validation"]$end_date[1])
    
    # Check if validation data exists for this country
    if(nrow(top_iso3_plot_dat[run_period == "validation"])>0){
      # Create epidemiological curves plot for validation period
      epi_curves <- ggplot(top_iso3_plot_dat[run_period == "validation"]) +
        # Observed data line
        geom_line(aes(y = y, x = ds, color = "Observed")) +
        # Model prediction lines
        geom_line(aes(y = yhat1, x = ds, group = MODEL_ID, color = MODEL_ID, linetype = is_cluster_run), alpha = 0.7) +
        # Vertical lines for cutoff and end dates
        geom_vline(xintercept = cutoff_num, color = "firebrick") +
        geom_vline(xintercept = end_num, color = "firebrick") +
        # Color scale for models
        scale_color_manual(
          values = c(
            "Observed" = "black",
            setNames(
              viridis::viridis(min(n_to_take, length(model_ids))),
              as.character(model_ids[seq_len(min(n_to_take, length(model_ids)))])
            )
          ),
          breaks = c("Observed", as.character(model_ids)),
          labels = c("Observed", paste0("MODEL_ID:", model_ids))) +
        # Line type scale (solid for individual runs, dashed for cluster runs)
        scale_linetype_manual(
          values = c("no" = "solid", "yes" = "dashed"),
          guide = "none"
        ) +
        # Labels and title
        labs(
          title = top_iso3_plot_dat$ID[1],
          subtitle = paste(top_iso3_plot_dat$model[1], "\n", top_iso3_plot_dat$predictor[1]),
          x = "Date", y = "Cases per 1M", color = "Series"
        ) +
        theme_minimal() +
        theme(
          legend.position = "bottom",
          plot.title = element_text(size = 12, hjust = 0.5),
          plot.subtitle = element_text(size = 8, hjust = 0.5)
        )
    }else{
      # Return empty plot with message if no validation data available
      print("No validation runs present for this country")
      epi_curves <- plotly::plot_ly() %>%
        plotly::layout(
          xaxis = list(showticklabels = FALSE),
          yaxis = list(showticklabels = FALSE),
          annotations = list(
            text = "No validation runs present for this country",
            x = 0.5, y = 0.5,
            xref = "paper", yref = "paper",
            showarrow = FALSE,
            font = list(size = 16)
          )
        )
    }
    
    return(epi_curves)
    
  })
  
  # Render binary outcome heatmap for model validation period
  # Shows predicted vs observed outbreak classifications for validation data
  output$binary_outcome_plots_validation <- renderPlotly({
    
    COUNTRY <- input$COUNTRY
    if(is.na(input$top_n))
      return
    
    # Get top N models based on user selection
    n_to_take <- min(input$top_n, nrow(summaryTable[ID == COUNTRY,]))
    top_iso3_plot_dat <- get_plot_dat(summ_dt = summaryTable, cutoff_dat = cutoff_dat,
                                      iso3 = COUNTRY, col_name = input$order_summary_col_by, n = n_to_take, by_config = "yes")
    
    # Generate binary outcome heatmap for validation period
    binary_outcome_analysis <- plot_binary_outcome(plot_dat = top_iso3_plot_dat, period = "validation")
    
    return(binary_outcome_analysis)
    
  })
  
}) # End of shinyServer function

