# Load necessary packages
library(shiny)
library(dplyr)
library(ggplot2)
library(readr)
library(shinythemes)
library(shinydashboard)


# Import US Census data
df_raw <- read.csv("C:/Users/kanem/Documents/Demographic_Insights/Data/acs2017_census_tract_data.csv")
# Remove missing values
census_data <- na.omit((df_raw))

# Define the user interface
ui <- dashboardPage(
  dashboardHeader(title = "County demographics"),
  dashboardSidebar(
    # Add input fields for selecting state and county
    selectInput(inputId = "state", label = "Select State:", choices = unique(census_data$State)),
    selectizeInput(inputId = "county", label = "Select County:", choices = NULL, multiple = TRUE)
  ),
  dashboardBody(
    fluidRow(
      # Add output for displaying average income
      box(title = "Average Income", solidHeader = TRUE, 
          plotOutput(outputId = "income_plot"),
          verbatimTextOutput(outputId = "mean_income_text")
      ),
      # Add output for displaying unemployment vs poverty levels
      box(title = "Unemployment vs Poverty Levels", solidHeader = TRUE, 
          plotOutput(outputId = "unemployment_plot")),
      # Add output for displaying MeanCommute vs Transportation Types
      box(title = "Average Commute Times", solidHeader = TRUE,
          plotOutput(outputId = "meancommute_plot"))
    )
  )
)

# Define the server function
server <- function(input, output, session) {
  # Update county choices based on selected state
  observeEvent(input$state, {
    county_choices <- unique(census_data$County[census_data$State == input$state])
    updateSelectizeInput(session, "county", choices = county_choices)
  })
  
  # Generate plot of average income
  output$income_plot <- renderPlot({
    filtered_data <- census_data %>% 
      filter(State == input$state, County %in% input$county)
    
    # Calculate mean income by county
    mean_income <- filtered_data %>% 
      group_by(County) %>% 
      summarise(mean_income = mean(Income)) %>% 
      arrange(mean_income)
    
    # Create bar plot of mean income by county
    ggplot(mean_income, aes(x = reorder(County, mean_income), y = mean_income)) +
      geom_point(stat = "identity", fill = "#0072B2", size=5,shape=15,alpha = 0.8) +  # blue fill color with transparency
      ggtitle("Average Income by County") +
      xlab("County") +
      ylab("Avg. Income") +
      theme_minimal() +  # minimal theme with white background
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +  # rotate x-axis labels and adjust position
      theme(plot.title = element_text(size = 14, face = "bold")) +  # increase title size and bold font
      labs(title = "Average Income by County", subtitle = paste("State:", input$state, "County(s):", paste(input$county, collapse = ", "))) +  # add title and subtitle
      scale_y_continuous(labels = scales::dollar_format(prefix="$"))
    
  })
  
  # Generate plot of unemployment vs poverty levels
  output$unemployment_plot <- renderPlot({
    filtered_data <- census_data %>% 
      filter(State == input$state, County == input$county)
    
    # Calculate average unemployment and poverty rates for the county
    avg_unemp <- mean(filtered_data$Unemployment)
    avg_pov <- mean(filtered_data$Poverty)
    
    # Create scatter plot of unemployment vs poverty levels
    ggplot(filtered_data, aes(x = Poverty, y = Unemployment, color=filtered_data$Income)) +
      geom_point() +
      geom_text(x = 20, y = 10, label = paste0("Avg. Unemployment Rate: ", round(avg_unemp, 2),"%", "\nAvg. Poverty Rate: ", round(avg_pov, 2),"%"), 
                hjust = 0, vjust = 1, size = 5, color = "black") +
      ggtitle("Unemployment vs Poverty Levels") +
      xlab("Poverty") +
      ylab("Unemployment")+
      labs(color = "Household Income")
  })
  
  # Generate plot of mean commute vs transportation types
  output$meancommute_plot <- renderPlot({
    filtered_data <- census_data %>% 
      filter(State == input$state, County == input$county)
    
    # Create scatter plot of mean commute vs transportation types
    ggplot(filtered_data, aes(x = MeanCommute)) +
      geom_density(fill = "blue", alpha = 0.2)
  })
}

# Run the app
shinyApp(ui, server)
