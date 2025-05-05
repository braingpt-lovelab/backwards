# Set working directory
setwd("/Users/ken/Desktop/LoveLab_desktop/_postdoc/BrainGPT/backwards/analyses")

# Load required package
library(lme4)

# Read the data
data <- read.csv("model_performance_x_direction_x_size_x_item.csv")

# Convert to factors
data$direction <- as.factor(data$direction)
data$model_id <- as.factor(data$model_id)

# Run repeated measures ANOVA
model <- aov(correct ~ direction*model_size + Error(model_id), data = data)

# Print the results
summary(model)