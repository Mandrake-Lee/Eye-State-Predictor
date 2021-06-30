# This script will source from internet the raw data
library(foreign)

# Downloading the data & importing to R
dataurl <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"

df <- tempfile()
download.file(dataurl, destfile=df, method="auto")

# Converting to a dataframe
rawData <- read.arff(df)

rawFile <- file.path(getwd(),"rda","rawData.rda")

# Saving for future use
save(rawData, file=rawFile)

# Cleaning
rm(dataurl, df, rawFile)