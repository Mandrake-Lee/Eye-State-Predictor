# Eye State Predictor
# Author: Jorge Amorós-Argos
# Date:   June 2021
# Language: R
#
# This project stems from the Data Science course, HarvardX-PH125.9x course
# as the ending project needed for submission 


message("==== Welcome to the Eye State Predictor ====")
# First check if "tidy" data is available below /rda
# This is the file that we need for our training
rawFile <- file.path(getwd(),"rda","rawData.rda")

if (!file.exists(rawFile)){
  source("Eye_State_Predictor_Sourcing.R")
} else {
  message("[OK] Rawdata file is ready. Loading...")
  load(rawFile)
  }

rm(rawFile)

# Let's load needed libraries
library(tidyverse)
library(caret)

# Let's load functions
source("Eye_State_Predictor_Functions.R")

# We know the total span of data is 117s; therefore the discrete step
h <- 117/nrow(rawData)

# After exploring the data (see report), we have to clean some points
eyeData <- rawData %>% mutate(avg=apply(.[1:14],1,mean),stdev= apply(.[1:14],1,sd)) %>%
          filter(avg<10000 & avg > 4200)

# Numerate chunks in time (consecutive rows) where eye is resting open/close
# We're cheating here because first value na is known to be "0"
eyeData <- eyeData %>% mutate(test=(eyeDetection!=lag(eyeDetection,default="0")), chunk=cumsum(test)) %>%
          select(-test)

message("[OK] Rawdata cleaned as eyeData.")

# Preparing the new dataset based on FFT exploration
# We will keep per each chunk/window:
# amp_0 (mean per window)
# Top 3 most representative frequencies f_1, f_2, f_3 (Hz)
eyeDataFFT <- eyeData %>% group_by(chunk) %>%
  summarise(eyeDetection = first(eyeDetection),
            across(1:14, ~ fft_top3(.,1/h)[1,2], .names="{.col}_a_0"),
            across(1:14, ~ fft_top3(.,1/h)[2,1], .names="{.col}_f_1"),
            across(1:14, ~ fft_top3(.,1/h)[3,1], .names="{.col}_f_2"),
            across(1:14, ~ fft_top3(.,1/h)[4,1], .names="{.col}_f_3"),
  )

message("[OK] eyeData transformed to fourier series as eyeDataFFT.")

# Because we have a very small dataset (only 24 time series)
# Let's simplify the same data
eyeDataFFTsimple <- eyeData %>% group_by(chunk) %>%
  summarise(eyeDetection = first(eyeDetection),
            across(1:14, ~ fft_top3(.,1/h)[1,2], .names="{.col}_avg"),
            across(1:14, ~ sum(fft_top3(.,1/h)[2:4,1]), .names="{.col}_freq")
  ) %>%
  pivot_longer(!c(eyeDetection,chunk), names_to=c("EEG","feature"), names_sep="_", values_to="Val") %>%
  unique() %>%
  group_by(chunk,feature,eyeDetection) %>%
  summarise(mean=mean(Val)) %>%
  pivot_wider(names_from="feature", values_from="mean") %>%
    ungroup()

message("[OK] eyeDataFFT reduced to eyeDataFFTsimple.")


### Starting the model training
message("[OK] Ready to start training")

set.seed(2007)
test_index <- createDataPartition(eyeDataFFTsimple$eyeDetection, times=1, p=0.10, list=FALSE)

#Create train & test sets
eyeDataFFTsimple_train <- eyeDataFFTsimple %>% select(-chunk) %>% slice(-test_index)
eyeDataFFTsimple_test <- eyeDataFFTsimple %>%  slice(test_index) %>% select(-chunk)

# Setting crossvalidation parameters
control <- trainControl(method="cv", number=10, p=0.90)

## Using regression
aux <- eyeDataFFTsimple_train %>% mutate(p=ifelse(eyeDetection=="0",0,1))
train_lm <- train(p ~ avg+freq, method = "lm",
                  data=aux,
                  trControl = control)
# Check accuracy
y_hat_lm <- predict(train_lm, eyeDataFFTsimple_test, type="raw")

y_hat_lm <- y_hat_lm %>% round() %>% factor(labels=c("0","1"))

aux <- confusionMatrix(y_hat_lm, eyeDataFFTsimple_test$eyeDetection)$overall["Accuracy"]
accuracy_results <- data.frame(method="LM", Accuracy = aux)


## Using knn
train_knn <- train(eyeDetection ~., method = "knn",
                   data=eyeDataFFTsimple_train, 
                   tuneGrid=data.frame(k=seq(1,11,2)),
                   trControl = control)
# Print K opt
train_knn$bestTune

# Check accuracy
y_hat_knn <- predict(train_knn, eyeDataFFTsimple_test, type="raw")

aux <- confusionMatrix(y_hat_knn, eyeDataFFTsimple_test$eyeDetection)$overall["Accuracy"]

accuracy_results <- bind_rows(accuracy_results,
                              data.frame(method="KNN", Accuracy = aux))

## Using QDA
train_qda <- train(eyeDetection ~., method = "qda",
                   data=eyeDataFFTsimple_train, 
                   trControl = control)
# Check accuracy
y_hat_qda <- predict(train_qda, eyeDataFFTsimple_test)

aux <- confusionMatrix(y_hat_qda, eyeDataFFTsimple_test$eyeDetection)$overall["Accuracy"]

accuracy_results <- bind_rows(accuracy_results,
                              data.frame(method="QDA", Accuracy = aux))

## Ensemble
ensemble <- as.matrix(rbind(y_hat_lm, y_hat_knn, y_hat_qda))

# Preparing ensemble
y_hat_ensemble <- ensemble %>% sweep(1,1) %>%
  colMeans() %>%
  round() %>%
  factor(labels=c("0","1"))

aux <- confusionMatrix(y_hat_ensemble, eyeDataFFTsimple_test$eyeDetection)$overall["Accuracy"]

accuracy_results <- bind_rows(accuracy_results,
                              data.frame(method="LM+KNN+QDA", Accuracy = aux))

### Print on screen the results of each model
print(knitr::kable(accuracy_results))

### Last not least, save relevant data in order to knit the R report
resultFile <- file.path(getwd(),"rda","eyePredictor.rda")
save(eyeDataFFTsimple, accuracy_results, train_lm, train_knn, train_qda, file=resultFile)

message("\n[OK] Done. Relevant results saved in 'rda/eyePredictor.rda'\n")

