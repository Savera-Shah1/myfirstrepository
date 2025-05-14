# =========================== R SCRIPT =========================================
# =========================== Group-08 =========================================
# =========================== FINAL PROJECT ====================================

#Loading Required Packages:
install.packages("ggplot2")
install.packages("reshape2")
install.packages("ggcorrplot")
install.packages("Metrics")
install.packages("tseries")
install.packages("forecast")
install.packages("GGally")
install.packages("TSA")
install.packages("mFilter")

#Loading necessary libraries
library(ggplot2)
library(forecast)
library(tseries)
library(dplyr)
library(tidyr)
library(Metrics)
library(TSA)
library(reshape2)
library(ggcorrplot)
library(lmtest)
library(GGally)
library(mFilter)

# =========================== Part 1 ===========================================
# =========================== INTRODUCTION =====================================

#Part1:
# Reading the data and creating time series object: -----------------
# Reading the Data. 
df <- read.csv("Electricity Price Data.csv")

# Convert to time series (January 1990 to November 2017, monthly frequency)
elec_ts <- ts(df$ElecPrice, start=c(1990, 1), frequency=12)

# Time Series Plots of All Variables.
# Create a multivariate time series object for all variables
mymts <- ts(df[, c("ElecPrice", "Electricity_Generation", "CPI", "NGAS")], 
            start = c(1990, 1), frequency = 12)

# Plot the time series
autoplot(mymts, facets = TRUE) +  # `facets = TRUE` plots each variable separately
  ggtitle("Time Series Plot of Electricity Pricing Dataset") +
  xlab("Year") + ylab("Values") +
  theme(plot.title = element_text(hjust = 0.5))  # Center title

# Box Plots of All variables 
# Reshape the data to long format for faceting 
df_long <- df %>%
  pivot_longer(cols = c("ElecPrice", "Electricity_Generation", "CPI", "NGAS"), 
               names_to = "Variable", values_to = "Value")

# Boxplots for all variables in a grid layout
ggplot(df_long, aes(x = factor(Month), y = Value)) +
  geom_boxplot(fill = "lightblue") +
  facet_wrap(~ Variable, scales = "free_y") +  # Creates separate panels for each variable
  ggtitle("Boxplots of Electricity Pricing Dataset Variables by Month") +
  xlab("Month") + ylab("Values") +
  theme(plot.title = element_text(hjust = 0.5))  # Center title

# Univariate Analysis: Boxplots, ACF/PACF Plots----------------
# Univariate time series plot of the variable of interest (i.e. Electricity Price)
autoplot(elec_ts) + ggtitle("Monthly Electricity Price in US") + ylab("ElecPrice")

# Boxplot for ElecPrice by month
# Seasonal Boxplots (By Month)
ggplot(df, aes(x=factor(Month), y=ElecPrice)) +
  geom_boxplot(fill="lightblue") +
  ggtitle("Seasonality in Electricity Price") +
  xlab("Month") + ylab("Electricity Price")

# Decomposition of the series: 
# Decompose the series (additive)
elec_decomp <- decompose(elec_ts, type = "additive")
plot(elec_decomp)

# ACF and PACF plots
acf(elec_ts, main="ACF of Electricity Price")
pacf(elec_ts, main="PACF of Electricity Price")

# ADF test for stationarity
adf_test <- adf.test(elec_ts)
print(adf_test)  

#============================== Part 2 =========================================
#============================== Univariate Time-series models ==================
#------------------------------ Part 2.1 & 2.1.1 -------------------------------
 # Deterministic Time Series Models --- Seasonal +Trend----------------------
# Create time index
df$Time <- 1:nrow(df)

# Convert Month to a factor variable
df$Month <- as.factor(df$Month)

# Split into training and hold-out sample
train_size <- 300
holdout_size <- 35

train_data <- df[1:train_size, ]
holdout_data <- df[(train_size+1):(train_size+holdout_size), ]

# Fit linear, quadratic, and cubic models on training data
linear_model <- lm(ElecPrice ~ Time + Month, data=train_data)
quadratic_model <- lm(ElecPrice ~ poly(Time, 2, raw=TRUE) + Month, data=train_data)
cubic_model <- lm(ElecPrice ~ poly(Time, 3, raw=TRUE) + Month, data=train_data)
quartic_model <- lm(ElecPrice ~ poly(Time, 4, raw=TRUE) + Month, data=train_data)
summary(linear_model)
summary(quadratic_model)
summary(cubic_model)
summary(quartic_model)

# Predict on hold-out sample for all models
holdout_data$Pred_Linear <- predict(linear_model, newdata=holdout_data)
holdout_data$Pred_Quadratic <- predict(quadratic_model, newdata=holdout_data)
holdout_data$Pred_Cubic <- predict(cubic_model, newdata=holdout_data)
holdout_data$Pred_quartic <- predict(quartic_model, newdata=holdout_data)


# Compare models using R-squared, MAPE, RMSE, and MAE
compute_metrics <- function(actual, predicted) {
  data.frame(
    RMSE = rmse(actual, predicted),
    MAE = mae(actual, predicted),
    MAPE = mean(abs((actual - predicted) / actual)) * 100,
    R_Squared = summary(lm(predicted ~ actual))$r.squared
  )
}

# Calculate metrics for each model
linear_metrics <- compute_metrics(holdout_data$ElecPrice, holdout_data$Pred_Linear)
quadratic_metrics <- compute_metrics(holdout_data$ElecPrice, holdout_data$Pred_Quadratic)
cubic_metrics <- compute_metrics(holdout_data$ElecPrice, holdout_data$Pred_Cubic)
quartic_metrics <- compute_metrics(holdout_data$ElecPrice, holdout_data$Pred_quartic)


# Combine results into one data frame
model_comparison <- bind_rows(
  Linear = linear_metrics,
  Quadratic = quadratic_metrics,
  Cubic = cubic_metrics,
  Quartic = quartic_metrics,
  .id = "Model Hold-Out Sample"
)

# Print model comparison Hold-out sample comparison.
print(model_comparison)

# Make predictions on training data
train_data$Pred_Linear <- predict(linear_model, newdata=train_data)
train_data$Pred_Quadratic <- predict(quadratic_model, newdata=train_data)
train_data$Pred_Cubic <- predict(cubic_model, newdata=train_data)
train_data$Pred_Quartic <- predict(quartic_model, newdata=train_data)

# Compute metrics for each model on training sample
linear_metrics_train <- compute_metrics(train_data$ElecPrice, train_data$Pred_Linear)
quadratic_metrics_train <- compute_metrics(train_data$ElecPrice, train_data$Pred_Quadratic)
cubic_metrics_train <- compute_metrics(train_data$ElecPrice, train_data$Pred_Cubic)
quartic_metrics_train <- compute_metrics(train_data$ElecPrice, train_data$Pred_Quartic)

# Combine training metrics into one data frame
train_model_comparison <- dplyr::bind_rows(
  Linear = linear_metrics_train,
  Quadratic = quadratic_metrics_train,
  Cubic = cubic_metrics_train,
  Quartic = quartic_metrics_train,
  .id = "Model Training Sample"
)

# Print training sample comparison
print(train_model_comparison)


# Actual Vs Predicted Plots of Linear, Quadratic, and Quartic Models. 
#1. Linear Model
# Create a combined dataset with predictions for both training and holdout samples
df$Pred_Linear <- predict(linear_model, newdata=df)  # Predict using the linear model
# Plot Actual vs Predicted for the entire dataset (training + holdout)
ggplot(df, aes(x=Time)) +
  geom_line(aes(y=ElecPrice, color="Actual"), size=1) +  # Actual values
  geom_line(aes(y=Pred_Linear, color="Linear"), linetype="dashed", size=1) +  # Fitted linear model
  ggtitle("Actual vs Predicted (Linear Model) for All Data") +
  ylab("ElecPrice") +
  xlab("Time") +
  scale_color_manual(values=c("black", "red")) +
  theme_minimal()

#2. Quadratic Model
# Create a combined dataset with predictions for both training and holdout samples
df$Pred_quadratic <- predict(quadratic_model, newdata=df)  # Predict using the quadratic model
# Plot Actual vs Predicted for the entire dataset (training + holdout)
ggplot(df, aes(x=Time)) +
  geom_line(aes(y=ElecPrice, color="Actual"), size=1) +  # Actual values
  geom_line(aes(y=Pred_quadratic, color="Quadratic"), linetype="dashed", size=1) +  # Fitted quadratic model
  ggtitle("Actual vs Predicted (Quadratic Model) for All Data") +
  ylab("ElecPrice") +
  xlab("Time") +
  scale_color_manual(values=c("black", "red")) +
  theme_minimal()

#3. Quartic Model
# Create a combined dataset with predictions for both training and holdout samples
df$Pred_quartic <- predict(quartic_model, newdata=df)  # Predict using the quartic model
# Plot Actual vs Predicted for the entire dataset (training + holdout)
ggplot(df, aes(x=Time)) +
  geom_line(aes(y=ElecPrice, color="Actual"), size=1) +  # Actual values
  geom_line(aes(y=Pred_quartic, color="Quartic"), linetype="dashed", size=1) +  # Fitted quartic model
  ggtitle("Actual vs Predicted (Quartic Model) for All Data") +
  ylab("ElecPrice") +
  xlab("Time") +
  scale_color_manual(values=c("black", "red")) +
  theme_minimal()


# Choice of the Best Model based on training and hold-out sample performance. 
## CUBIC MODEL---------------------------------------------------------------
# Actual Vs Predicted 
# Create a combined dataset with predictions for both training and holdout samples
df$Pred_Cubic <- predict(cubic_model, newdata=df)  # Predict using the cubic model

# Plot Actual vs Predicted for the entire dataset (training + holdout)
ggplot(df, aes(x=Time)) +
  geom_line(aes(y=ElecPrice, color="Actual"), size=1) +  # Actual values
  geom_line(aes(y=Pred_Cubic, color="Cubic"), linetype="dashed", size=1) +  # Fitted cubic model
  ggtitle("Actual vs Predicted (Cubic Model) for All Data") +
  ylab("ElecPrice") +
  xlab("Time") +
  scale_color_manual(values=c("black", "red")) +
  theme_minimal()

# Performance Comparison of Training Vs Hold-out sample of cubic model. 
# Define a function to compute performance metrics
compute_metrics <- function(actual, predicted) {
  data.frame(
    RMSE = rmse(actual, predicted),
    MAE = mae(actual, predicted),
    MAPE = mean(abs((actual - predicted) / actual)) * 100,
    R_Squared = summary(lm(predicted ~ actual))$r.squared
  )
}
# Compute metrics for the training sample
train_data$Pred_Cubic <- predict(cubic_model, newdata=train_data)  # Predictions on training data
train_metrics <- compute_metrics(train_data$ElecPrice, train_data$Pred_Cubic)

# Compute metrics for the holdout sample
holdout_data$Pred_Cubic <- predict(cubic_model, newdata=holdout_data)  # Predictions on holdout data
holdout_metrics <- compute_metrics(holdout_data$ElecPrice, holdout_data$Pred_Cubic)

# Combine and display the metrics for both training and holdout samples
comparison <- data.frame(
  Sample = c("Training", "Holdout"),
  RMSE = c(train_metrics$RMSE, holdout_metrics$RMSE),
  MAE = c(train_metrics$MAE, holdout_metrics$MAE),
  MAPE = c(train_metrics$MAPE, holdout_metrics$MAPE),
  R_Squared = c(train_metrics$R_Squared, holdout_metrics$R_Squared)
)

# Print the comparison table
print(comparison)

# residuals
# Get residuals from cubic model
residuals_cubic <- residuals(cubic_model)

# ACF and PACF plots of residuals
acf(residuals_cubic, main="ACF of Residuals (Cubic Model)")
pacf(residuals_cubic, main="PACF of Residuals (Cubic Model)")

# ADF test for residuals' stationarity (Not Stationary)
adf_residuals <- adf.test(residuals_cubic)
print(adf_residuals)  

# Ljung-Box test for autocorrelation in residuals 
#(If the p-value is greater than 0.05, you can assume that the residuals are white noise.)
Box.test(residuals_cubic, lag=20, type="Ljung-Box")


#------------------------------------3-Piece Trend Model------------------------------------
# Defining dummies
dummy1 <- rep(0, length(elec_ts)) # Jan 2001 to Dec 2008
dummy2 <- rep(0, length(elec_ts)) #After Dec 2008

# Assign dummy values based on time segments
for (i in 1:length(elec_ts)) {
  if (df$Time[i] >= 113 & df$Time[i] < 220) {
    dummy1[i] <- 1
  } else {
    dummy1[i] <- 0
  }
  
  if (df$Time[i] > 219) {
    dummy2[i] <- 1
  } else {
    dummy2[i] <- 0
  }
}

# Creating interaction terms
int1 <- dummy1 * df$Time
int2 <- dummy2 * df$Time

# Adding dummy and interaction terms to original data
df$dummy1 <- dummy1
df$dummy2 <- dummy2
df$int1 <- int1
df$int2 <- int2

# Plot electricity prices with vertical lines at breakpoints
ggplot(df, aes(x = Time, y = ElecPrice)) +
  geom_line(color = "black", size = 1) +
  geom_vline(xintercept = c(113, 220), linetype = "dashed", color = "red", size = 1) +
  annotate("text", x = 60, y = max(df$ElecPrice), label = "Segment 1", color = "blue", size = 4) +
  annotate("text", x = 160, y = max(df$ElecPrice), label = "Segment 2", color = "blue", size = 4) +
  annotate("text", x = 250, y = max(df$ElecPrice), label = "Segment 3", color = "blue", size = 4) +
  ggtitle("Electricity Price with 3-Piece Trend Segments") +
  xlab("Time") + ylab("Electricity Price") +
  theme_minimal()

# Spliting the data-set into train and hold-out samples.
train_data <- df[1:train_size, ]
holdout_data <- df[(train_size+1):(train_size+holdout_size), ]

# Fit 3-piece segmented trend model
three_piece_model <- lm(ElecPrice ~ Time + dummy1 + dummy2 + int1 + int2 + Month, data = train_data)
summary(three_piece_model)

#Predictions
train_data$Pred_3Piece <- predict(three_piece_model, newdata = train_data)
holdout_data$Pred_3Piece <- predict(three_piece_model, newdata = holdout_data)

# Performance Evaluation
train_metrics_3piece <- compute_metrics(train_data$ElecPrice, train_data$Pred_3Piece)
holdout_metrics_3piece <- compute_metrics(holdout_data$ElecPrice, holdout_data$Pred_3Piece)
comparison_3piece <- data.frame(
  Sample = c("Training", "Holdout"),
  RMSE = c(train_metrics_3piece$RMSE, holdout_metrics_3piece$RMSE),
  MAE = c(train_metrics_3piece$MAE, holdout_metrics_3piece$MAE),
  MAPE = c(train_metrics_3piece$MAPE, holdout_metrics_3piece$MAPE),
  R_Squared = c(train_metrics_3piece$R_Squared, holdout_metrics_3piece$R_Squared)
)
print(comparison_3piece)

# Merge training and holdout data back together
combined_df <- bind_rows(train_data, holdout_data)

# Plot Actual vs Predicted for 3-piece trend model
ggplot(combined_df, aes(x = Time)) +
  geom_line(aes(y = ElecPrice, color = "Actual"), size = 1) +
  geom_line(aes(y = Pred_3Piece, color = "Predicted (3-Piece Model)"), linetype = "dashed", size = 1) +
  ggtitle("Actual vs Predicted Electricity Prices - 3-Piece Trend Model") +
  ylab("Electricity Price") +
  xlab("Time") +
  scale_color_manual(values = c("Actual" = "black", "Predicted (3-Piece Model)" = "red")) +
  theme_minimal()

# Detrending the series from the 3-piece model
train_data$Detrended_3Piece <- residuals(three_piece_model)

# boxplot using the training data only:
ggplot(train_data, aes(x = Month, y = Detrended_3Piece)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  ggtitle("Boxplot of Detrended Electricity Price by Month (3-Piece Model)") +
  xlab("Month") + ylab("Detrended Electricity Price") +
  theme_minimal()

# ACF and PACF plots of residuals from 3-piece model
acf(residuals(three_piece_model), main = "ACF of Residuals (3-Piece Model)", lag.max=96)
pacf(residuals(three_piece_model), main = "PACF of Residuals (3-Piece Model)", lag.max=96)

#-----------------------Cyclical Model--------------------------------------------
#-a. Log Transformation Model-----------------------------------------------------
lelec_ts=log(elec_ts)
n_elec_ts=lelec_ts[1:300]

# plot of log Prices
plot.ts(n_elec_ts,col="blue",ylab="log Prices")

time=seq(1,length(elec_ts))

# Periodogram for series 
library(TSA)
prdgrm=periodogram(fit$residuals,col="blue")
period=1/prdgrm$freq

par(mfrow=c(1,2))
periodogram(fit$residuals,col="blue")
plot(period,prdgrm$spec, type="h",col="blue",ylab="Peridogram",lwd=2)

# Creating the Cyclical Terms. (Except sin6 because sin6 = 0)

time=seq(1,length(n_elec_ts))

sin1=sin(2*pi*(1/12)*time)
cos1=cos(2*pi*(1/12)*time)

sin2=sin(2*pi*(2/12)*time)
cos2=cos(2*pi*(2/12)*time)

sin3=sin(2*pi*(3/12)*time)
cos3=cos(2*pi*(3/12)*time)

sin4=sin(2*pi*(4/12)*time)
cos4=cos(2*pi*(4/12)*time)

cos5=cos(2*pi*(5/12)*time)
sin5=sin(2*pi*(5/12)*time)

cos6=cos(2*pi*(6/12)*time)


# fitting the model using harmonics 1-6 components
fit<-lm(n_elec_ts~time+sin1+cos1+sin2+cos2+sin3+cos3+sin4+cos4+sin5+cos5+cos6)
summary(fit)
plot.ts(elec_ts[1:300], type="b",col="blue",ylab="Electricity Price",lwd=2)
lines(exp(predict(fit)),col="red",lwd=2)
acf(residuals(fit))

#Compare with Seasonal Indicators
fit2<-lm(n_elec_ts~time+as.factor(df$Month[1:300]))
summary(fit2)
plot.ts(elec_ts[1:300], type="b",col="blue")
lines(exp(predict(fit2)),col="red")

# Training Predictions
train_pred = exp(predict(fit))

# Training Sample Performance 
MAPE_train = mean(abs(elec_ts[1:300] - train_pred) / elec_ts[1:300]) * 100
MAPE_train
RMSE_train = sqrt(mean((elec_ts[1:300] - train_pred)^2))
RMSE_train
MAE_train = mean(abs(elec_ts[1:300] - train_pred))
MAE_train

#Prediction (Hold-out Sample)
pred = exp(predict(fit, data.frame(
  time = c(301:335),
  sin1 = sin(2 * pi * (1/12) * c(301:335)),
  cos1 = cos(2 * pi * (1/12) * c(301:335)),
  sin2 = sin(2 * pi * (2/12) * c(301:335)),
  cos2 = cos(2 * pi * (2/12) * c(301:335)),
  sin3 = sin(2 * pi * (3/12) * c(301:335)),
  cos3 = cos(2 * pi * (3/12) * c(301:335)),
  sin4 = sin(2 * pi * (4/12) * c(301:335)),
  cos4 = cos(2 * pi * (4/12) * c(301:335)),
  sin5 = sin(2 * pi * (5/12) * c(301:335)),
  cos5 = cos(2 * pi * (5/12) * c(301:335)),
  cos6 = cos(2 * pi * (6/12) * c(301:335))
)))

# Hold-out Sample Performance 
MAPE=mean( abs(elec_ts[301:335]-pred)/elec_ts[301:335] ) * 100
MAPE
RMSE = sqrt(mean((elec_ts[301:335] - pred)^2))
RMSE
MAE = mean(abs(elec_ts[301:335] - pred))
MAE

# Comparison Table
comparison = matrix(c(MAPE_train, RMSE_train, MAE_train,
                      MAPE,       RMSE,       MAE),
                    nrow = 3, byrow = FALSE)
rownames(comparison) = c("MAPE", "RMSE", "MAE")
colnames(comparison) = c("Training", "Holdout")
comparison

#b. Detrended Series.
# Periodogram for Electricity Price Series 
#removing trend
detrend<-lm(n_elec_ts~time)

prdgrm=periodogram(detrend$residuals,col="blue")

period=1/prdgrm$freq

par(mfrow=c(1,2))
periodogram(detrend$residuals,col="blue")
plot(period,prdgrm$spec, type="h",col="blue",ylab="Peridogram",lwd=2)

# Listing periodogram values

frequency=prdgrm$freq
amplitude=prdgrm$spec


all=cbind(period,frequency,amplitude)
all

all_sorted <- all[order(-all[,3]),]
top8 <- head(all_sorted, 8)
head(top8, n=8)

# Creating the sine and cosine terms for the periods (300, 12, 150, 100, 6,75,30, & 60) 
#--the numbers are  in the order of highest to lowest. 
n=length(n_elec_ts)

cos1=cos(2*pi*(1/n)*time)
sin1=sin(2*pi*(1/n)*time)

cos2=cos(2*pi*(2/n)*time)
sin2=sin(2*pi*(2/n)*time)

cos3=cos(2*pi*(3/n)*time)
sin3=sin(2*pi*(3/n)*time)

cos4=cos(2*pi*(4/n)*time)
sin4=sin(2*pi*(4/n)*time)

cos5=cos(2*pi*(5/n)*time)
sin5=sin(2*pi*(5/n)*time)

cos10=cos(2*pi*(10/n)*time)
sin10=sin(2*pi*(10/n)*time)

cos25=cos(2*pi*(25/n)*time)
sin25=sin(2*pi*(25/n)*time)

cos50=cos(2*pi*(50/n)*time)
sin50=sin(2*pi*(50/n)*time)

#estimation of the model

fit2<-lm(n_elec_ts~time+cos1+sin1+cos2+sin2+cos3+sin3+cos4+sin4+cos5+sin5+cos10+sin10+
           cos25+sin25+cos50+sin50)
summary(fit2)

# Actual Vs Predicted
plot.ts(elec_ts[1:n], type = "b", col = "blue", ylab = "Electricity Price", lwd = 2)
lines(exp(predict(fit2)), col = "red", lwd = 2)

# Training Predictions Detrended Series
train_pred = exp(predict(fit2))

# Training Sample Performance Detrended Series
MAPE_train = mean(abs(elec_ts[1:300] - train_pred) / elec_ts[1:300]) * 100
MAPE_train
RMSE_train = sqrt(mean((elec_ts[1:300] - train_pred)^2))
RMSE_train
MAE_train = mean(abs(elec_ts[1:300] - train_pred))
MAE_train

#Prediction for months 301-335
time_n=c(301:335)

cos1_n=cos(2*pi*(1/n)*time_n)
sin1_n=sin(2*pi*(1/n)*time_n)

cos2_n=cos(2*pi*(2/n)*time_n)
sin2_n=sin(2*pi*(2/n)*time_n)

cos3_n=cos(2*pi*(3/n)*time_n)
sin3_n=sin(2*pi*(3/n)*time_n)

cos4_n=cos(2*pi*(4/n)*time_n)
sin4_n=sin(2*pi*(4/n)*time_n)

cos5_n=cos(2*pi*(5/n)*time_n)
sin5_n=sin(2*pi*(5/n)*time_n)

cos10_n=cos(2*pi*(10/n)*time_n)
sin10_n=sin(2*pi*(10/n)*time_n)

cos25_n=cos(2*pi*(25/n)*time_n)
sin25_n=sin(2*pi*(25/n)*time_n)

cos50_n=cos(2*pi*(50/n)*time_n)
sin50_n=sin(2*pi*(50/n)*time_n)


pred=exp(predict(fit2,data.frame(time=time_n,cos1=cos1_n,sin1=sin1_n, cos2=cos2_n,sin2=sin2_n,cos3=cos3_n,sin3=sin3_n,
                                 cos4=cos4_n,sin4=sin4_n, cos5=cos5_n,sin5=sin5_n,cos10=cos10_n,
                                 sin10=sin10_n, cos25=cos25_n,sin25=sin25_n,cos50=cos50_n,sin50=sin50_n)))               
# Hold-out Sample Performance 
MAPE=mean( abs(elec_ts[301:335]-pred)/elec_ts[301:335] ) * 100
MAPE
RMSE = sqrt(mean((elec_ts[301:335] - pred)^2))
RMSE
MAE = mean(abs(elec_ts[301:335] - pred))
MAE

# Comparison Table
comparison = matrix(c(MAPE_train, RMSE_train, MAE_train,
                      MAPE,       RMSE,       MAE),
                    nrow = 3, byrow = FALSE)
rownames(comparison) = c("MAPE", "RMSE", "MAE")
colnames(comparison) = c("Training", "Holdout")
comparison
acf(residuals(fit2))

#-------------------------------Part 2.2 & 2.2.1 -------------------------------

elec_ts <- ts(df$ElecPrice, start=c(1990, 1), frequency=12)

  # 2.2 Exponential Smoothing models (only the relevant ones).
  elec_ts

train_elec <- elec_ts[1:300]
test_elec <- elec_ts[301:335]
length(train_elec)  # Training data (1990-2014)
length(test_elec) # Hold-out data (Jan 2015 - Nov 2017)


# Define Hold-out Period (301-335 -> Jan 2015 - Nov 2017)
window_elec=window(elec_ts,start=c(1990, 1),end=c(2014, 12))
hw_es=hw(window_elec,h=1)
summary(hw_es)
# Holt Winters Model MAPE
pred_HW <- fitted(hw_es)

HW_pred=rep(0,335)
HW_pred[1:300]=pred_HW[1:300]
HW_pred[1:300]

# Obtaining the level and slope estimates for training set
L_t=rep(0,335)
T_t=rep(0,335)
S_t=rep(0,335)

# Parameters and Initials from the output
HW_alpha=hw_es$model$par["alpha"]
HW_beta=hw_es$model$par["beta"]
HW_gamma=hw_es$model$par["gamma"]

HW_L_0=hw_es$model$states[1, "l"]
HW_T_0=hw_es$model$states[1, "b"]

print(hw_es$model$states[1, 3:14])

HW_S_0 = c(-0.261954596, -0.233114594, 0.023154624, 0.211733634, 0.332610070, 
           0.442391618, 0.276573715, -0.009185287, -0.164162668, -0.167847299, 
           -0.211616820, -0.238582397)
S_t[1:12] = HW_S_0

# Updating the First separately

L_t[1]=HW_alpha*(train_elec[1]-HW_S_0[1])+(1-HW_alpha)*(HW_L_0+HW_T_0)
T_t[1]=HW_beta*(L_t[1]-HW_L_0)+(1-HW_beta)*HW_T_0
S_t[1]=HW_gamma*(train_elec[1]-L_t[1])+(1-HW_gamma)*S_t[1]

# Updating the 2nd to 12th separately as instructed

for (t in 2:12) {
  L_t[t] = HW_alpha * (train_elec[t] - S_t[t]) + (1 - HW_alpha) * (L_t[t-1] + T_t[t-1])
  T_t[t] = HW_beta * (L_t[t] - L_t[t-1]) + (1 - HW_beta) * T_t[t-1]
  S_t[t] = HW_gamma * (train_elec[t] - L_t[t]) + (1 - HW_gamma) * S_t[t]
}

# Updating 12th to 300th separately after 1 to 12 are done

for (t in 13:300){
  L_t[t]=HW_alpha*(train_elec[t]-S_t[t-12])+(1-HW_alpha)*(L_t[t-1]+T_t[t-1])
  T_t[t]=HW_beta*(L_t[t]-L_t[t-1])+(1-HW_beta)*T_t[t-1]
  S_t[t]=HW_gamma*(train_elec[t]-L_t[t])+(1-HW_gamma)*S_t[t-12]
}

for (t in 301:335) {  
  # ONE-STEP AHEAD FORECASTS FOR THE HOLD-OUT SAMPLE
  HW_pred[t] = L_t[t-1] + T_t[t-1] + S_t[t-12]
  
  # UPDATING FOR THE HOLD-OUT SAMPLE
  L_t[t] = HW_alpha * (elec_ts[t] - S_t[t-12]) + (1 - HW_alpha) * (L_t[t-1] + T_t[t-1])
  T_t[t] = HW_beta * (L_t[t] - L_t[t-1]) + (1 - HW_beta) * T_t[t-1]
  S_t[t] = HW_gamma * (elec_ts[t] - L_t[t]) + (1 - HW_gamma) * S_t[t-12]
}

HW_mape_hold=mean(abs(elec_ts[301:335]-HW_pred[301:335])/elec_ts[301:335])
cat("Holt Winters Exponential Smoothing MAPE:", HW_mape_hold*100, "%\n")


# Train set prediction
# Calculating MAPE, RMSE and MAE
train_mape_hw <- mean(abs(window_elec - HW_pred[1:300]) / window_elec) * 100
train_rmse_hw <- sqrt(mean((window_elec - HW_pred[1:300])^2))
train_mae_hw <- mean(abs(window_elec - HW_pred[1:300]))

results <- data.frame(
  Metric = c("MAPE", "RMSE", "MAE"),
  Value = c(train_mape_hw, train_rmse_hw, train_mae_hw)
)

cat("\n--- Training Holt-Winters Model Results ---\n")
print(results)


# Hold out sample/Test set prediction 
# calculating MAPE, RMSE and MAE for Test
mape_hw <- mean(abs(elec_ts[301:335]-HW_pred[301:335])/elec_ts[301:335]) * 100
rmse_hw <- sqrt(mean((elec_ts[301:335]-HW_pred[301:335])^2))
mae_hw <- mean(abs(elec_ts[301:335]-HW_pred[301:335]))

results2 <- data.frame(
  Metric = c("MAPE", "RMSE", "MAE"),
  Value = c(mape_hw, rmse_hw, mae_hw)
)

cat("\n--- Test Holt-Winters Model Results ---\n")
print(results2)


#============================== Part 3 =========================================
#============================== Time Series Regression Models ==================
#-------------------------------Part 3.1, 3.2, & 3.3 ---------------------------
# Load dataset
df <- read.csv("C:/Users/hzx20/OneDrive/桌面/Electricity Price Data (1).csv")

# Select relevant variables
data <- df[, c("ElecPrice", "Electricity_Generation", "NGAS", "CPI")]

# ---- Correlation Matrix ----
correlation_matrix <- cor(data, use = "complete.obs")
print("Correlation Matrix:")
print(round(correlation_matrix, 2))

# Optional: Heatmap
ggcorr(data, label = TRUE, label_round = 2, label_size = 4, hjust = 0.75, layout.exp = 1)

# ---- Scatter Plots ----
par(mfrow = c(1, 3))
plot(data$Electricity_Generation, data$ElecPrice, col = "orange",
     main = "Electricity Price vs. Electricity Generation",
     xlab = "Electricity Generation", ylab = "Electricity Price")
plot(data$NGAS, data$ElecPrice, col = "orange",
     main = "Electricity Price vs. NGAS",
     xlab = "NGAS", ylab = "Electricity Price")
plot(data$CPI, data$ElecPrice, col = "orange",
     main = "Electricity Price vs. CPI",
     xlab = "CPI", ylab = "Electricity Price")

# Create train/test split (last 36 rows for test)
n <- nrow(df)
train_df <- df[1:(n - 36), ]
test_df <- df[(n - 35):n, ]

y_train <- train_df$ElecPrice
y_test <- test_df$ElecPrice

# Candidate models
candidate_models <- list(
  "Full Model" = c("Electricity_Generation", "NGAS", "CPI"),
  "Reduced Model 1" = c("NGAS", "CPI"),
  "Reduced Model 2" = c("CPI")
)

# Metric function
get_metrics <- function(y_true, y_pred, include_r2 = FALSE) {
  mape_val <- mape(y_true, y_pred) * 100
  rmse_val <- rmse(y_true, y_pred)
  mae_val <- mae(y_true, y_pred)
  metrics <- list(MAPE = mape_val, RMSE = rmse_val, MAE = mae_val)
  if (include_r2) {
    r2_val <- summary(lm(y_true ~ y_pred))$r.squared
    metrics$R2 <- r2_val
  }
  return(metrics)
}

# Run and collect results
results <- data.frame()

for (name in names(candidate_models)) {
  predictors <- candidate_models[[name]]
  
  model <- lm(ElecPrice ~ ., data = train_df[, c("ElecPrice", predictors)])
  train_preds <- predict(model, train_df)
  test_preds <- predict(model, test_df)
  
  train_metrics <- get_metrics(y_train, train_preds, include_r2 = TRUE)
  test_metrics <- get_metrics(y_test, test_preds)
  
  results <- rbind(results, data.frame(
    Model = name,
    Train_R2 = train_metrics$R2,
    Train_MAPE = train_metrics$MAPE,
    Train_RMSE = train_metrics$RMSE,
    Train_MAE = train_metrics$MAE,
    Test_MAPE = test_metrics$MAPE,
    Test_RMSE = test_metrics$RMSE,
    Test_MAE = test_metrics$MAE
  ))
}

# Round only numeric columns
results_rounded <- results
numeric_cols <- sapply(results, is.numeric)
results_rounded[numeric_cols] <- round(results[numeric_cols], 3)

print(results_rounded)

# Fit model
model <- lm(ElecPrice ~ Electricity_Generation + NGAS + CPI, data = train_df)

train_preds <- predict(model, train_df)
test_preds <- predict(model, test_df)

# Evaluation
get_metrics <- function(y_true, y_pred) {
  list(
    R2 = summary(lm(y_true ~ y_pred))$r.squared,
    MAPE = mape(y_true, y_pred) * 100,
    RMSE = rmse(y_true, y_pred),
    MAE = mae(y_true, y_pred)
  )
}

train_metrics <- get_metrics(y_train, train_preds)
test_metrics <- get_metrics(y_test, test_preds)

print("Training Metrics:")
print(train_metrics)
print("Test Metrics:")
print(test_metrics)

results_table <- data.frame(
  Metric = names(train_metrics),
  Train = unlist(train_metrics),
  Test = unlist(test_metrics),
  row.names = NULL
)

print(results_table)

# Residuals
residuals <- residuals(model)

# ADF Test
adf_result <- adf.test(residuals)
print(paste("ADF Test p-value:", adf_result$p.value))

# Ljung-Box Test (lag 12)
lb_result <- Box.test(residuals, lag = 12, type = "Ljung-Box")
print("Ljung-Box Test (lag 12):")
print(lb_result)

# Plot residuals and ACF
par(mfrow = c(1, 2))
plot(residuals, type = "l", main = "Time Series Plot of Residuals")
abline(h = 0, col = "gray", lty = 2)

acf(residuals, main = "ACF of Residuals")

#===========================  Part - 4 =========================================
# =========================== Stochastic Time Series Models ====================
#-------------------------------- Part 4.1 -------------------------------------
# ARIMA models (for the variable of interest)

# Reading the Data. 
df <- read.csv("Electricity Price Data.csv")

# Convert to time series (January 1990 to November 2017, monthly frequency)
elec_ts <- ts(df$ElecPrice, start=c(1990, 1), frequency=12)
elec_ts
length(elec_ts)
# Training set: 1 to 300 observations (Jan 1990 to Dec 2014)
elec_train <- ts(window(elec_ts, end = c(2014, 12)))
length(elec_train)

# Test set: 301 to 335 observations (Jan 2015 to Nov 2017)
elec_test <- ts(window(elec_ts, start = c(2015, 1)))
length(elec_test)
elec_test

#TS plot of Train Series
autoplot(elec_train) + ggtitle("Monthly Electricity Price in US") + ylab("ElecPrice")
# Boxplot for ElecPrice by month
# Seasonal Boxplots (By Month)
ggplot(df, aes(x=factor(Month), y=ElecPrice)) +
  geom_boxplot(fill="lightblue") +
  ggtitle("Seasonality in Electricity Price") +
  xlab("Month") + ylab("Electricity Price")
# Decomposition of the series: 
# Decompose the series (additive)
elec_decomp <- decompose(elec_ts, type = "additive")
plot(elec_decomp)
# ACF and PACF plots of Train Series
library(gridExtra)
p1 <- ggAcf((elec_train), lag.max = 36) + ggtitle("ACF")
p2 <- ggPacf((elec_train), lag.max = 36) + ggtitle("PACF")
grid.arrange(p1, p2, ncol = 2)


# Inducing Seasonality
par(mfrow=c(1,2))
q1 <- ggAcf((elec_train),lag=36,col="blue", main = "Price")
q2 <- ggAcf(diff((elec_train)),lag=36,col="blue", main = "Price - ACF Non-Seasonal Difference") + geom_vline(xintercept = seq(12, 48, 12), linetype="dashed", color="red")
q3 <- ggAcf(diff(diff((elec_train)), lag=12),lag=36,col="blue", main = "Price - ACF Seasonal Difference") 
q4 <- ggPacf(diff(diff((elec_train)), lag=12),lag=36,col="blue",ylim=c(-0.5,1), main = "Price - PACF Seasonal Difference")
grid.arrange(q1, q2,q3, q4, ncol = 2) 
plot(diff(diff(elec_train), lag = 12), main = "Double Differenced (d=1, D=1)", ylab = "Differenced Price", col = "blue")
# checking the auto arima: 
auto.arima(elec_ts)
#We can specify the series by including Box-Cox transformation parameter lambda=0
sarima.fit <- Arima(elec_train, order = c(0, 1, 0), seasonal=list(order=c(0,1,1), period=12))
summary(sarima.fit)
acf(sarima.fit$residuals)

# hold out sample accuracy
hold=elec_test

Preds=Arima(elec_ts,model=sarima.fit)
summary(Preds)
# accuracy measures for hold out sample
holdfit=fitted(Preds)
accuracy(hold,holdfit[301:335])



#-------------------------------- Part 4.2 -------------------------------------
# Read the data
df <- read.csv("Electricity Price Data.csv")

# Create time and seasonal variables
df$Time <- 1:nrow(df)
df$Month <- as.factor(df$Month)

# =============== SECTION 4.2 - Regression Residual SARIMA =====================

# Fit regression model
reg_model <- lm(ElecPrice ~ Electricity_Generation + NGAS + CPI, data = df)

# Extract residuals
res <- residuals(reg_model)

# --- ACF and PACF of Original Residuals
par(mfrow=c(1,2))
acf(res, lag.max=36, col="blue", main="ACF of Residuals")
pacf(res, lag.max=36, col="blue", main="PACF of Residuals")
par(mfrow=c(1,1))

# --- Stationarity and White Noise Tests
adf.test(res)
Box.test(res, lag=24, type="Ljung-Box")

# --- Differencing: Non-seasonal and seasonal (lag 1 + lag 12)
res_diff <- diff(diff(res), lag=12)

par(mfrow=c(1,2))
acf(res_diff, lag.max=36, col="darkgreen", main="ACF of (1-B^12)(1-B) Residuals")
pacf(res_diff, lag.max=36, col="darkgreen", main="PACF of (1-B^12)(1-B) Residuals")
par(mfrow=c(1,1))

# --- Fit SARIMA model to residuals
fit_res_sarima <- Arima(res, order=c(0,1,1), seasonal=list(order=c(0,1,1), period=12))
summary(fit_res_sarima)

# --- Residual diagnostics of SARIMA
acf(fit_res_sarima$residuals, col="red", main="ACF of SARIMA Residuals")
Box.test(fit_res_sarima$residuals, lag=24, type="Ljung-Box")

#============ SECTION 4.2 EXTENSION - ARIMA with XREG ==========================
# Prepare regressors as external variables
xreg_matrix <- cbind(df$Electricity_Generation, df$NGAS, df$CPI)
colnames(xreg_matrix) <- c("Generation", "NGAS", "CPI")

# Fit ARIMA model with regression + AR(2) errors
regar <- Arima(df$ElecPrice, order = c(2, 0, 0), xreg = xreg_matrix)
summary(regar)
#-------------------------------- Part 4.3 -------------------------------------
# ============= SECTION 4.3 - Deterministic Model Residuals ====================

# Fit Cubic Trend + Seasonality model
cubic_model <- lm(ElecPrice ~ poly(Time, 3, raw = TRUE) + Month, data = df)
cubic_residuals <- residuals(cubic_model)

# --- Time Series Plot
plot.ts(cubic_residuals, main = "Time Series Plot of Cubic Model Residuals", ylab = "Residuals", col = "darkgreen")

# --- ACF & PACF
par(mfrow=c(1,2))
acf(cubic_residuals, main="ACF of Residuals (Cubic Model)")
pacf(cubic_residuals, main="PACF of Residuals (Cubic Model)")
par(mfrow=c(1,1))

# --- Tests
adf_cubic <- adf.test(cubic_residuals)
cat("ADF Test p-value (Cubic Residuals):", adf_cubic$p.value, "\n")
ljung_cubic <- Box.test(cubic_residuals, lag = 20, type = "Ljung-Box")
cat("Ljung-Box p-value (Cubic Residuals):", ljung_cubic$p.value, "\n")

# --- HP Filter
hp_cycle <- hpfilter(cubic_residuals, freq = 14400)
plot.ts(hp_cycle$cycle, main = "Cyclical Component from HP Filter (Cubic Model Residuals)", col = "purple", ylab = "Cycle")

# =========== SECTION 4.3 EXTENSION - 3-Piece Trend Model=====================

# Define breakpoints (manual inspection)
break1 <- 100
break2 <- 200

# Create piecewise time variables
df$seg1 <- ifelse(df$Time <= break1, df$Time, break1)
df$seg2 <- ifelse(df$Time > break1 & df$Time <= break2, df$Time - break1, 0)
df$seg3 <- ifelse(df$Time > break2, df$Time - break2, 0)

# Fit 3-piece segmented trend model
seg_model <- lm(ElecPrice ~ poly(seg1, 3, raw = TRUE) + poly(seg2, 3, raw = TRUE) + poly(seg3, 3, raw = TRUE) + Month, data = df)
seg_resid <- residuals(seg_model)

# --- Plot residuals
plot.ts(seg_resid, main = "Time Series Plot of Segmented Trend Model Residuals", ylab = "Residuals", col = "blue")

# --- ACF & PACF
par(mfrow = c(1,2))
acf(seg_resid, main = "ACF of Residuals (3-Piece Trend Model)")
pacf(seg_resid, main = "PACF of Residuals (3-Piece Trend Model)")
par(mfrow = c(1,1))

# --- Tests
adf_seg <- adf.test(seg_resid)
cat("ADF Test p-value (Segmented Residuals):", adf_seg$p.value, "\n")
ljung_seg <- Box.test(seg_resid, lag = 20, type = "Ljung-Box")
cat("Ljung-Box p-value (Segmented Residuals):", ljung_seg$p.value, "\n")

