
############################################################
# Load Libraries
############################################################
library(tidyverse)
library(ggplot2)
library(ggrepel)
library(patchwork)
library(caret)
library(knitr)
library(kableExtra)
library(rpart)
#install.packages("rpart.plot")
library(rpart.plot)
library(randomForest)
library(pdp)
library(dplyr)
#install.packages("lime")
library(lime)
library(scales)

############################################################
# Import Data
############################################################
df <- read.csv("company_esg_financial_dataset.csv")

# Data overview
str(df)
summary(df)

############################################################
# Basic Cleaning
############################################################

# Drop identifiers
df <- df %>% 
  select(-CompanyID, -CompanyName)

# Convert categorical variables to factors
df <- df %>%
  mutate(
    Industry = as.factor(Industry),
    Region   = as.factor(Region),
    Year     = as.factor(Year)
  )

############################################################
# Handle Missing Values
############################################################

sum(is.na(df$GrowthRate))

# Replace missing GrowthRate values with median
df$GrowthRate[is.na(df$GrowthRate)] <- median(df$GrowthRate, na.rm = TRUE)

############################################################
# Exploratory Data Analysis
############################################################

# Visualize distributions of numeric variables
num_vars <- df %>% select(where(is.numeric))

par(mfrow = c(3,4))
for (v in names(num_vars)) {
  hist(num_vars[[v]],
       main = v,
       xlab = "",
       breaks = 30)
}
par(mfrow = c(1,1))

############################################################
# Feature Engineering
############################################################

# Log-transform highly skewed variables
df <- df %>%
  mutate(
    log_MarketCap         = log1p(MarketCap),
    log_CarbonEmissions   = log1p(CarbonEmissions),
    log_WaterUsage        = log1p(WaterUsage),
    log_EnergyConsumption = log1p(EnergyConsumption)
  )

# Create ESG classification target
df <- df %>%
  mutate(
    ESG_Class = ifelse(
      ESG_Overall > median(ESG_Overall, na.rm = TRUE),
      "High",
      "Low"
    ),
    ESG_Class = as.factor(ESG_Class)
  )

############################################################
# Data for Clustering
############################################################

env_vars <- df %>%
  select(
    ESG_Environmental,
    log_CarbonEmissions,
    log_WaterUsage,
    log_EnergyConsumption
  )

env_scaled <- scale(env_vars)

############################################################
# Data for Supervised Models
############################################################

model_data <- df %>%
  select(
    ESG_Class,
    Revenue,
    ProfitMargin,
    GrowthRate,
    log_MarketCap,
    Industry,
    Region,
    Year,
    log_CarbonEmissions,
    log_WaterUsage,
    log_EnergyConsumption
  )

############################################################
# Train/Test Split
############################################################

set.seed(123)

train_index <- sample(
  seq_len(nrow(model_data)),
  size = 0.7 * nrow(model_data)
)

train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]

############################################################
# Hierarchical Clustering
############################################################

# Distance matrix (Euclidean)
dist_env <- dist(env_scaled, method = "euclidean")

hc_env <- hclust(dist_env, method = "ward.D2")

plot(
  hc_env,
  labels = FALSE,
  main = "Hierarchical Clustering Dendrogram",
  xlab = "",
  ylab = "Height"
)

# Cut dendrogram into 3 clusters
cluster_labels <- cutree(hc_env, k = 3)

# Add cluster labels to main data
df$Cluster <- as.factor(cluster_labels)

table(df$Cluster)

# Cluster Profiles
df %>%
  group_by(Cluster) %>%
  summarise(
    `Carbon Emissions`        = mean(CarbonEmissions, na.rm = TRUE),
    `Water Usage`             = mean(WaterUsage, na.rm = TRUE),
    `Energy Consumption`      = mean(EnergyConsumption, na.rm = TRUE),
    `Environmental ESG Score` = mean(ESG_Environmental, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  kable(
    format = "latex",
    booktabs = TRUE,
    digits = 1,
    align = c("l", "r", "r", "r", "r"),
    caption = "\\textbf{Environmental Sustainability Cluster Profiles}"
  ) %>%
  kable_styling(
    full_width = FALSE,
    position = "center",
    latex_options = c("hold_position")
  ) 

# Label clusters
df$Cluster_label <- factor(
  df$Cluster,
  levels = c("1", "2", "3"),
  labels = c("Sustainable", "Moderate impact", "High impact")
)


# Common color scale
cluster_colors <- c(
  "Sustainable"     = "forestgreen",
  "Moderate impact" = "orange",
  "High impact"     = "black"
)

# Small text setting
small_text <- theme(
  plot.title   = element_text(size = 9),
  axis.title   = element_text(size = 7),
  axis.text    = element_text(size = 7),
  legend.title = element_text(size = 8),
  legend.text  = element_text(size = 7)
)

# Plot 1: raw observations
plot_1 <- ggplot(df, aes(
  x = log_WaterUsage,
  y = log_CarbonEmissions,
  color = Cluster_label )
) +
  geom_point(alpha = 0.2, size = 0.6) +
  scale_color_manual(
    name   = "Cluster",
    values = cluster_colors
  ) +
  labs(title = "Figure 1: Environmental Sustainability Clusters",
       x = "Log Water Usage",
       y = "Log Carbon Emissions") +
  guides(color = guide_legend(override.aes = list(alpha = 1, size = 3))) + theme_minimal() +
  theme(plot.title = element_text(face = "bold"))+
  small_text


# Cluster averages (for plot_2)
cluster_summary <- df %>%
  group_by(Cluster_label) %>%
  summarise(
    log_water  = mean(log_WaterUsage, na.rm = TRUE),
    log_carbon = mean(log_CarbonEmissions, na.rm = TRUE),
    .groups = "drop"
  )

# Plot 2: cluster averages 
plot_2 <- ggplot(cluster_summary, aes(
  x = log_water,
  y = log_carbon,
  color = Cluster_label
)
) +
  geom_point(size = 6) +
  scale_color_manual(
    values = cluster_colors,
    guide  = "none"   
  ) +
  coord_cartesian(
    xlim = range(cluster_summary$log_water) + c(-0.3, 0.3),
    ylim = range(cluster_summary$log_carbon) + c(-0.3, 0.3)
  ) +
  labs(title = "Figure 2: Average Environmental Clusters",
       x = "Average Water Usage",
       y = "Average Carbon Emissions") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))+
  small_text

# Combine plots
final_plot <- (plot_1 | plot_2) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
final_plot


############################################################
# Logistic Regression (Baseline Model)
############################################################

logit_model <- glm(
  ESG_Class ~ Revenue + ProfitMargin + GrowthRate +
    log_MarketCap + Industry + Region + Year +
    log_CarbonEmissions + log_WaterUsage + log_EnergyConsumption,
  data = train_data,
  family = binomial
)

summary(logit_model)

# predict on test data
logit_pred_prob <- predict(
  logit_model,
  newdata = test_data,
  type = "response"
)

logit_pred_class <- ifelse(logit_pred_prob > 0.5, "High", "Low")
logit_pred_class <- as.factor(logit_pred_class)

# Evaluate Performance

# confusion matrix
cm_logit <- confusionMatrix(
  logit_pred_class,
  test_data$ESG_Class,
  positive = "High"
)
print(cm_logit)

# Visualize Confusion Matrix
cm_df <- as.data.frame(cm_logit$table)
colnames(cm_df) <- c("Predicted", "Actual", "Count")

# Heatmap
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), size = 5) +
  scale_fill_gradient(
    low = "lightblue",
    high = "darkseagreen3"
  ) +
  labs(
    title = "Confusion Matrix Heatmap – Logistic Regression",
    x = "Actual ESG Class",
    y = "Predicted ESG Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 9),
    axis.title = element_text(size = 10),
    axis.text  = element_text(size = 9)
  )

#Train predictions
logit_train_prob <- predict(
  logit_model,
  newdata = train_data,
  type = "response"
)

logit_train_class <- ifelse(logit_train_prob > 0.5, "High", "Low") %>%
  as.factor()

# Confusion Matrix for train data
cm_train <- confusionMatrix(
  logit_train_class,
  train_data$ESG_Class,
  positive = "High"
)

#Test predictions
logit_test_prob <- predict(
  logit_model,
  newdata = test_data,
  type = "response"
)

logit_test_class <- ifelse(logit_test_prob > 0.5, "High", "Low") %>%
  as.factor()

# Confusion Matrix for test data
cm_test <- confusionMatrix(
  logit_test_class,
  test_data$ESG_Class,
  positive = "High"
)

# Heatmap
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), size = 5) +
  scale_fill_gradient(
    low = "lightblue",
    high = "darkseagreen3"
  ) +
  labs(
    title = "Confusion Matrix Heatmap – Logistic Regression",
    x = "Actual ESG Class",
    y = "Predicted ESG Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 9),
    axis.title = element_text(size = 10),
    axis.text  = element_text(size = 9)
  )


# Summary statistics
stats_logit <- data.frame(
  `Train Accuracy`      = cm_train$overall["Accuracy"],
  `Test Accuracy`       = cm_test$overall["Accuracy"],
  Kappa               = cm_test$overall["Kappa"],
  Sensitivity         = cm_test$byClass["Sensitivity"],
  Specificity         = cm_test$byClass["Specificity"],
  `Balanced Accuracy`   = cm_test$byClass["Balanced Accuracy"],
  row.names = NULL
)

# Table for logistic regression 
kable(stats_logit, digits = 3,
      caption = "\\textbf{Performance Metrics for Logistic Regression}")%>%
  kable_styling(full_width = FALSE, position = "center")


############################################################
# Decision Tree
############################################################

tree_model <- rpart(
  ESG_Class ~ Revenue + ProfitMargin + GrowthRate +
    log_MarketCap + Industry + Region + Year +
    log_CarbonEmissions + log_WaterUsage + log_EnergyConsumption,
  data = train_data,
  method = "class",
  control = rpart.control(
    cp = 0.01,        # complexity parameter (prevents overfitting)
    minsplit = 20,    # minimum obs to split
    maxdepth = 6      # keeps tree interpretable
  )
)

# Visualize Tree
rpart.plot(
  tree_model,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Figure 3: Decision Tree for ESG Classification",
  cex.main = 1.15,   
  split.cex = 0.9)


# Train predictions
tree_train_class <- predict(
  tree_model,
  newdata = train_data,
  type = "class"
)

cm_tree_train <- confusionMatrix(
  tree_train_class,
  train_data$ESG_Class,
  positive = "High"
)

# Test predictions
tree_test_class <- predict(
  tree_model,
  newdata = test_data,
  type = "class"
)

cm_tree_test <- confusionMatrix(
  tree_test_class,
  test_data$ESG_Class,
  positive = "High"
)

# Visualize Confusion Matrix
cm_df_tree <- as.data.frame(cm_tree_test$table)
colnames(cm_df_tree) <- c("Predicted", "Actual", "Count")

# Heatmap
ggplot(cm_df_tree, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), size = 5) +
  scale_fill_gradient(
    low = "lightblue",
    high = "darkseagreen3"
  ) +
  labs(
    title = "Confusion Matrix Heatmap – Decision Tree",
    x = "Actual ESG Class",
    y = "Predicted ESG Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12),
    axis.title = element_text(size = 10),
    axis.text  = element_text(size = 9)
  )

# Summary statistics
stats_tree <- data.frame(
  `Train Accuracy`    = as.numeric(cm_tree_train$overall["Accuracy"]),
  `Test Accuracy`     = as.numeric(cm_tree_test$overall["Accuracy"]),
  Kappa             = as.numeric(cm_tree_test$overall["Kappa"]),
  Sensitivity       = as.numeric(cm_tree_test$byClass["Sensitivity"]),
  Specificity       = as.numeric(cm_tree_test$byClass["Specificity"]),
  `Balanced Accuracy` = as.numeric(cm_tree_test$byClass["Balanced Accuracy"])
)

# Table for decision tree
kable(stats_tree, digits = 3,
      caption = "\\textbf{Performance Metrics for Descision Tree}") %>%
  kable_styling(full_width = FALSE, position = "center")


############################################################
# Random Forest
############################################################

set.seed(123)

rf_model <- randomForest(
  ESG_Class ~ Revenue + ProfitMargin + GrowthRate +
    log_MarketCap + Industry + Region + Year +
    log_CarbonEmissions + log_WaterUsage + log_EnergyConsumption,
  data = train_data,
  ntree = 500,          # enough trees for stability
  mtry  = 3,            # default-ish for classification
  importance = TRUE
)

rf_model$err.rate[rf_model$ntree, "OOB"]

# Train predictions
rf_train_class <- predict(
  rf_model,
  newdata = train_data,
  type = "class"
)

cm_rf_train <- confusionMatrix(
  rf_train_class,
  train_data$ESG_Class,
  positive = "High"
)

# Test predictions
rf_test_class <- predict(
  rf_model,
  newdata = test_data,
  type = "class"
)

cm_rf_test <- confusionMatrix(
  rf_test_class,
  test_data$ESG_Class,
  positive = "High"
)

# Visualize Confusion Matrix
cm_df_rf <- as.data.frame(cm_rf_test$table)
colnames(cm_df_rf) <- c("Predicted", "Actual", "Count")

# Heatmap
p_rf <- ggplot(cm_df_rf, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), size = 5) +
  scale_fill_gradient(
    low = "lightblue",
    high = "darkseagreen3"
  ) +
  labs(
    title = "Figure 4: Confusion Matrix – Random Forest",
    x = "Actual ESG Class",
    y = "Predicted ESG Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 9, face = "bold"),
    axis.title = element_text(size = 8),
    axis.text  = element_text(size = 6)
  )

# Performance Metrics
stats_rf <- data.frame(
  'Train Accuracy'    = as.numeric(cm_rf_train$overall["Accuracy"]),
  'Test Accuracy'     = as.numeric(cm_rf_test$overall["Accuracy"]),
  Kappa             = as.numeric(cm_rf_test$overall["Kappa"]),
  Sensitivity       = as.numeric(cm_rf_test$byClass["Sensitivity"]),
  Specificity       = as.numeric(cm_rf_test$byClass["Specificity"]),
  'Balanced Accuracy' = as.numeric(cm_rf_test$byClass["Balanced Accuracy"])
)

# Table for random forest
kable(stats_rf, digits = 3,
      caption = "\\textbf{Performance Metrics for Random Forest}") %>%
  kable_styling(full_width = FALSE, position = "center")



############################################################
# Random Forest Tuned
############################################################

set.seed(123)

# X = predictors only, y = target factor
x_train <- train_data %>% select(-ESG_Class)
y_train <- train_data$ESG_Class

# Tune mtry using OOB error
rf_tune <- tuneRF(
  x = x_train,
  y = y_train,
  mtryStart = floor(sqrt(ncol(x_train))),  # sensible start
  stepFactor = 1.5,
  improve = 0.01,     # stop if improvement < 1%
  ntreeTry = 300,
  trace = TRUE,
  plot = TRUE
)

rf_tune
best_mtry <- rf_tune[which.min(rf_tune[, "OOBError"]), "mtry"]
best_mtry

# Fit final tuned Random Forest
set.seed(123)
rf_model_tuned <- randomForest(
  ESG_Class ~ .,
  data = train_data,
  ntree = 500,
  mtry = best_mtry,
  importance = TRUE
)

# Train
rf_tuned_train_class <- predict(rf_model_tuned, newdata = train_data, type = "class")
cm_rf_tuned_train <- confusionMatrix(rf_tuned_train_class, train_data$ESG_Class, positive = "High")

# Test
rf_tuned_test_class <- predict(rf_model_tuned, newdata = test_data, type = "class")
cm_rf_tuned_test <- confusionMatrix(rf_tuned_test_class, test_data$ESG_Class, positive = "High")

cm_rf_tuned_test


cm_df_rf_tuned <- as.data.frame(cm_rf_tuned_test$table)
colnames(cm_df_rf_tuned) <- c("Predicted", "Actual", "Count")

p_rf_tuned <- ggplot(cm_df_rf_tuned, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkseagreen3") +
  labs(
    title = "Figure 5: Confusion Matrix – Random Forest (Tuned)",
    x = "Actual ESG Class",
    y = "Predicted ESG Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 9, face = "bold"),
    axis.title = element_text(size = 8),
    axis.text  = element_text(size = 6)
  )


# Performance Metrics
stats_rf_tuned <- data.frame(
  'Train Accuracy'    = as.numeric(cm_rf_tuned_train$overall["Accuracy"]),
  'Test Accuracy'     = as.numeric(cm_rf_tuned_test$overall["Accuracy"]),
  Kappa             = as.numeric(cm_rf_tuned_test$overall["Kappa"]),
  Sensitivity       = as.numeric(cm_rf_tuned_test$byClass["Sensitivity"]),
  Specificity       = as.numeric(cm_rf_tuned_test$byClass["Specificity"]),
  'Balanced Accuracy' = as.numeric(cm_rf_tuned_test$byClass["Balanced Accuracy"])
)

# Table for tuned random forest
kable(stats_rf_tuned, digits = 3,
      caption = "\\textbf{Performance Metrics for Tuned Random Forest}") %>%
  kable_styling(full_width = FALSE, position = "center")

############################################################
# Model Performance of All Models
############################################################

model_performance <- rbind(
  Logistic_Regression = stats_logit,
  Decision_Tree       = stats_tree,
  Random_Forest       = stats_rf,
  Random_Forest_Tuned = stats_rf_tuned
)

kable(model_performance, digits = 3,
      caption = "\\textbf{Performance Metrics Comparison}") %>%
  kable_styling(full_width = FALSE, position = "center")


############################################################
# Global Interpretation
############################################################

# Variable Importance
importance_df <- importance(rf_model_tuned) %>%
  as.data.frame()

importance_df$Variable <- rownames(importance_df)

importance_df <- importance_df %>%
  arrange(desc(MeanDecreaseGini))

# Variable Importance Plot
ggplot(
  importance_df[1:10, ],
  aes(x = reorder(Variable, MeanDecreaseGini),
      y = MeanDecreaseGini)
) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Figure 6: Top 10 Variable Importance – Random Forest",
    x = "",
    y = "Mean Decrease in Gini"
  ) +
  theme_minimal()+
  theme(
    plot.title   = element_text(size = 9, face = "bold"),
    axis.title   = element_text(size = 7),
    axis.text    = element_text(size = 7),
    legend.title = element_text(size = 9),
    legend.text  = element_text(size = 7)
  )

# Partial Dependence Plots for Top 2 Variables

#PDP for Revenue
pdp_revenue <- partial(
  object = rf_model,
  pred.var = "Revenue",
  train = train_data,
  which.class = "High",
  prob = TRUE,
  grid.resolution = 30,
  plot = FALSE,
  parallel = FALSE  
)

p1 <- ggplot(pdp_revenue, aes(x = Revenue, y = yhat)) +
  geom_area(alpha = 0.15, fill = "steelblue") +
  geom_line(size = 1, color = "steelblue") +
  labs(
    title = "Figure 7: Partial Dependence of Revenue",
    x = "Revenue",
    y = "Predicted P(High ESG)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank()
  ) +
  theme(
    plot.title   = element_text(size = 9),
    axis.title   = element_text(size = 7),
    axis.text    = element_text(size = 7),
    legend.title = element_text(size = 9),
    legend.text  = element_text(size = 7)
  )

# Region bar plot 
rf_test_prob_high <- predict(rf_model_tuned, newdata = test_data, type = "prob")[, "High"]

region_effect <- test_data %>%
  mutate(P_High = rf_test_prob_high) %>%
  group_by(Region) %>%
  summarise(
    avg_prob_high = mean(P_High, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

p2 <- ggplot(region_effect, aes(x = reorder(Region, avg_prob_high), y = avg_prob_high)) +
  geom_col(fill = "steelblue", alpha = 0.9) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Figure 8: Average Predicted by Region",
    x = "Region",
    y = "Average Predicted P(High ESG)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank()
  ) +
  theme(
    plot.title   = element_text(size = 9),
    axis.title   = element_text(size = 7),
    axis.text    = element_text(size = 7),
    legend.title = element_text(size = 9),
    legend.text  = element_text(size = 7)
  )


# Side-by-side layout
(p1 | p2)

############################################################
# Local Interpretation
############################################################

set.seed(123)


# Define model_type and predict_model for randomForest
model_type.randomForest <- function(x, ...) {
  "classification"
}

predict_model.randomForest <- function(x, newdata, ...) {
  preds <- predict(x, newdata = newdata, type = "prob")
  as.data.frame(preds)
}

# Build LIME explainer on training predictors

x_train <- train_data %>% select(-ESG_Class)

rf_explainer <- lime(
  x = x_train,
  model = rf_model_tuned
)

# Pick one confident High and one confident Low example

rf_test_prob  <- predict(rf_model_tuned, newdata = test_data, type = "prob")
rf_test_class <- predict(rf_model_tuned, newdata = test_data, type = "class")

test_with_pred <- test_data %>%
  mutate(
    Predicted = rf_test_class,
    P_High    = rf_test_prob[, "High"]
  )

# Most confident predicted High
obs_high <- test_with_pred %>%
  filter(Predicted == "High") %>%
  arrange(desc(P_High)) %>%
  slice(1)

# Most confident predicted Low
obs_low <- test_with_pred %>%
  filter(Predicted == "Low") %>%
  arrange(P_High) %>%
  slice(1)


# Explain both observations with LIME

explain_one <- function(one_row, case_label) {
  lime::explain(
    x = one_row %>% select(-ESG_Class, -Predicted, -P_High),
    explainer = rf_explainer,
    labels = "High",          # explain probability of "High" ESG
    n_features = 6,
    n_permutations = 1000,
    kernel_width = 0.75
  ) %>%
    mutate(case = case_label)
}

exp_high <- explain_one(obs_high, "Example predicted High ESG")
exp_low  <- explain_one(obs_low,  "Example predicted Low ESG")

# Plot local explanations 
#high
plot_features(exp_high) +
  labs(title = "Figure 9: Local explanation - High ESG example") +
  theme_minimal() +
  theme(
    plot.title = element_text(
      size = 9,
      face = "bold",
      hjust = 0.5   
    ),
    axis.title = element_text(size = 7, face = "plain"),
    axis.text  = element_text(size = 7, face = "plain")
  )

#low
plot_features(exp_low) +
  labs(title = "Figure 10: Local explanation - Low ESG example") +
  theme_minimal() +
  theme(
    plot.title = element_text(
      size = 9,
      face = "bold",
      hjust = 0.5   
    ),
    axis.title = element_text(size = 7, face = "plain"),
    axis.text  = element_text(size = 7, face = "plain")
  )

```