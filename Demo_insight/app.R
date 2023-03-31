# Import necessary packages
library(shiny)
library(dplyr)
library(ggplot2)
library(readr)

# Import U.S. Census data (assuming it's stored in a CSV file)
census_data <- read.csv("C:/Users/kanem/Documents/Demographic_Insights/Data/acs2017_census_tract_data.csv")

# Define the user interface
ui <- fluidPage(
  # Add input fields for selecting demographic features and location
  selectInput(inputId = "race", label = "Race", choices = c("Hispanic", "White", "Black", "Native", "Asian", "Pacific")),
  numericInput(inputId = "income", label = "Minimum Income", value = 0),
  numericInput(inputId = "employment", label = "Minimum Employment Rate", value = 0),
  selectInput(inputId = "state", label = "State", choices = unique(census_data$State)),
  selectInput(inputId = "county", label = "County", choices = NULL),
  
  # Add a table and plot to display summary statistics
  tableOutput(outputId = "summary_table"),
  plotOutput(outputId = "summary_plot")
)

# Define the server function
server <- function(input, output) {
  # Update county choices based on selected state
  observeEvent(input$state, {
    county_choices <- unique(census_data$County[census_data$State == input$state])
    updateSelectInput(session, "county", choices = county_choices)
  })
  
  # Generate summary statistics based on selected demographic features and location
  summary_stats <- reactive({
    filtered_data <- census_data %>%
      filter(State == input$state, County == input$county, 
             get(input$race) > 0, Income >= input$income, Employed >= input$employment)
    
    # Calculate summary statistics
    mean_income <- mean(filtered_data$Income)
    pct_hispanic <- 100 * sum(filtered_data$Hispanic) / sum(filtered_data$TotalPop)
    pct_white <- 100 * sum(filtered_data$White) / sum(filtered_data$TotalPop)
    pct_black <- 100 * sum(filtered_data$Black) / sum(filtered_data$TotalPop)
    pct_native <- 100 * sum(filtered_data$Native) / sum(filtered_data$TotalPop)
    pct_asian <- 100 * sum(filtered_data$Asian) / sum(filtered_data$TotalPop)
    pct_pacific <- 100 * sum(filtered_data$Pacific) / sum(filtered_data$TotalPop)
    
    # Create a data frame with summary statistics
    summary_df <- data.frame(
      "Statistic" = c("Mean Income", "Percent Hispanic", "Percent White", "Percent Black", "Percent Native", "Percent Asian", "Percent Pacific"),
      "Value" = c(mean_income, pct_hispanic, pct_white, pct_black, pct_native, pct_asian, pct_pacific)
    )
    
    return(summary_df)
  })
  
  #
  