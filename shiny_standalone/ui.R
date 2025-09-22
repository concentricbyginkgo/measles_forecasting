############################################
# MEASLES MODEL VALIDATION SHINY APP - UI SCRIPT
# 
# This script defines the user interface for the measles model validation Shiny app.
# It creates a responsive web interface with:
# - Country selection dropdown
# - Model performance metrics table
# - Interactive epidemiological curve plots
# - Binary outcome prediction heatmaps
# - Separate views for model selection and validation periods
#
# Author: Ginkgo Biosecurity
# Date: September 17, 2025
# Version: 1.0
############################################

# Configure Shiny options for debugging and error handling
options(shiny.sanitize.errors = FALSE)
options(shiny.TRACE = TRUE)

# Define HTML dependencies for external libraries
# Bootstrap 4.3.1 for responsive layout and styling
bootstrapDep <- htmltools::htmlDependency("bootstrap", "4.3.1",
                                          src = c(href = "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/"),
                                          script = list(src = "js/bootstrap.min.js",
                                                        integrity = "sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM",
                                                        crossorigin = "anonymous"),
                                          stylesheet = list(src = "css/bootstrap.min.css"
                                          )
)

# Popper.js 1.14.7 for Bootstrap dropdown functionality
popperDep <- htmltools::htmlDependency("popper", "1.14.7",
                                       src = c(href = "https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/"),
                                       script = list(src = "umd/popper.min.js",
                                                     integrity = "sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1",
                                                     crossorigin = "anonymous"
                                       )
)

# jQuery 3.3.1 for JavaScript functionality
jQueryDep <- htmltools::htmlDependency("jquery", "3.3.1",
                                       src = c(href = "https://code.jquery.com/"),
                                       script = list(src = "jquery-3.3.1.slim.min.js",
                                                     integrity = "sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo",
                                                     crossorigin = "anonymous"
                                       )
)

# Font Awesome for icons
fontAwesomeDep <- htmltools::tagList(fontawesome::fa_html_dependency())

# Configuration variables for UI elements
status <- "default"
width = NULL
label = "Measles Model Selection + Validation + Forecasting"
# Define button appearance as a label (no dropdown functionality)
html_button <- list(
  class = paste0("btn btn-", status," shadow-none"),
  type = "button",
  style = "font-family:'sofia';color:#FFFFFF;"
)
html_button <- c(html_button, list(label))

# Main UI definition - combines all dependencies and page structure
ui = tagList(bootstrapDep, popperDep, fontAwesomeDep, jQueryDep,
             bootstrapPage(
               # Page head section with favicon and custom CSS
               tags$head(
                 tags$link(rel="icon", href="favicon.png", type="image/x-icon"),
                 tags$link(rel="shortcut icon", href="favicon.png"),
                 # includeCSS("www/bottom-sheet-drawer/downupPopup/downupPopup.css"),
                 includeCSS("www/styles.css"),
               ),
               # Main page body
               tags$body(
                 tags$div(class="container-fluid", style = "padding: 0px !important;",
                          # Main application page with custom styling
                          fluidPage(id = "app_page", style = "margin: 0px !important; padding: 0px !important;", tags$style(type = "text/css", ".row {margin-right: 0;}"),
                                    windowTitle = "Ginkgo Biosecurity Measles Validation",
                                    # Top navigation bar with company branding
                                    tags$nav(class="navbar navbar-expand navbar-dark bg-dark2", tags$style(type = "text/css", ".bg-dark2 {background-color: #1E1E3D !important;}"),
                                             # Company logo
                                             tags$a(href="#!", class="navbar-brand", tags$img(src="Concentric_logo_horizontal_color_white.png", style="margin-top: -3px;padding-right:30px;", height = "26")),
                                             # Navigation menu
                                             tags$div(class="navbar-collapse collapse justify-content-stretch", id="top_navbar",
                                                      # Main navigation tabs
                                                      tags$ul(class="navbar-nav", style = "border-bottom: 0px !important;",
                                                              tags$li(
                                                                role = "presentation", class="nav-item active px-auto",
                                                                tags$a(href="#model_evaluation", "data-target" = "#model_evaluation", "data-toggle" = "tab", "data-value" = "Model Evaluation", tags$div("Model Evaluation and Validation", style = "text-decoration: none;font-family:'sofia';color:#FFFFFF;") 
                                                                )
                                                              ),
                                                      ),
                                                      # User dropdown menu (right-aligned)
                                                      tags$ul(class="navbar-nav dropdown ml-auto",
                                                              do.call(tags$button, html_button),
                                                      )
                                             )
                                    ),
                                    # Main layout with sidebar and content area
                                    sidebarLayout(
                                      # Left sidebar panel for user controls
                                      sidebarPanel(width=2, 
                                                   # Custom styling for sidebar
                                                   style = "
                                                                  margin-left: 0 !important;
                                                                  margin-top: 0 !important;
                                                                  padding-left: 0 !important;
                                                                  height: calc(100vh - 60px) !important;  /* adjust 60px = navbar height */
                                                                  overflow-y: auto !important;
                                                                  overflow-x: hidden !important;
                                                          ",
                                                   # Additional CSS for form elements
                                                   tags$style(type = "text/css", ".col-sm-2 {background: #1E1E3D; padding: 0px !important; margin: 0px !important;}"),
                                                   tags$style(HTML("
                                                                        input[type=number] {
                                                                              -moz-appearance:textfield;
                                                                        }
                                                                        input[type=number]::{
                                                                              -moz-appearance:textfield;
                                                                        }
                                                                        input[type=number]::-webkit-outer-spin-button,
                                                                        input[type=number]::-webkit-inner-spin-button {
                                                                              -webkit-appearance: none;
                                                                              margin: 0;
                                                                        }

                                                                        input[type=number]{
                                                                            height: 30px;
                                                                            width: 90px;
                                                                        }

                                                                        select{
                                                                            height: 30px;
                                                                        }
                                                                        .selectize-input { text-align: left; font-size: 0.9em; font-family:sofia !important;}
                                                                        .selectize-dropdown { text-align: left; font-size: 0.9em; font-family:sofia !important;}

                                                                    ")),
                                                   
                                                   # Sidebar header
                                                   tags$h5("Selections", style = "padding-top: 10px;padding-left: 30px;font-size: 1em; font:sofia !important;color:#FFFFFF;"),
                                                   tags$hr(style = "border-top: 1px solid #FFFFFF;"),
                                                   
                                                   # Country selection dropdown
                                                   tags$div(align = 'left',
                                                            selectInput("COUNTRY",
                                                                        label = tags$span(class = "tblcell",
                                                                                          tags$span(class = "tblcell", style = "font-family:sofia !important;color:#FFFFFF;font-size:12;", "Select Country",
                                                                                                    tags$span(
                                                                                                      class = "tblcell fa fa-info-circle",
                                                                                                      title = paste0("Select Country")
                                                                                                    ))),
                                                                        choices = country_list, selected = "AFG"),
                                                            style = "padding-left: 30px; width:175px;font-family:sofia !important;color:#FFFFFF;font-size:12;"),
                                      ), # sidebarPanel
                                      
                                      # Main content panel
                                      mainPanel(width=10, tags$style(type = "text/css", ".col-sm-10 {padding: 0 !important; margin: 0 !important;}"),
                                                # Main tab content area
                                                tags$div(class="tab-content",
                                                         # Model Evaluation tab (active by default)
                                                         tags$div(class = "tab-pane active", "data-value" = "Model Evaluation", id = "model_evaluation",
                                                                  # Main content column
                                                                  column(
                                                                    12,
                                                                    tags$div(
                                                                      # Section separator
                                                                      tags$br(),
                                                                      tags$hr(style = "border-top: 1px solid #000000;"),
                                                                      tags$p('Select parameters'),
                                                                      
                                                                      # Parameter selection controls
                                                                      fluidRow(
                                                                        column(6,
                                                                               numericInput("top_n", "First n:", 5, min = 1, max = 100)),
                                                                        column(6,
                                                                               selectInput("order_summary_col_by",
                                                                                           "Order by: ",
                                                                                           choices = num_cols[!(num_cols %like% "R2")],
                                                                                           selected = "Test_MSE"))
                                                                      ),
                                                                      
                                                                      # Model performance metrics table
                                                                      dataTableOutput("model_evaluation_table"),
                                                                      
                                                                      # Model Selection section
                                                                      tags$h5('Model Selection'),
                                                                      plotlyOutput("epi_curves"),
                                                                      plotlyOutput("binary_outcome_plots"),
                                                                      
                                                                      # Model Validation section
                                                                      tags$h5('Model Validation'),
                                                                      plotlyOutput("epi_curves_validation"),
                                                                      plotlyOutput("binary_outcome_plots_validation")
                                                                    )
                                                                  ),
                                                         ),
                                                )
                                      ) # mainPanel
                                      
                                    ) # side bar layout
                                    
                          )
                          
                 )
               ) # end body
             ))
