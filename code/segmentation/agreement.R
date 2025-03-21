library(tidyverse)
library(seewave)

## Gaussian Smooth Data
smooth_responses <- function(time_series, bandwidth) {
  
  smoothed <- ksmooth(x = 0:(length(time_series) - 1), time_series, 'normal', bandwidth)
  return(smoothed$y)
}

## Agreement Index

agreement_index = function(sample, group){
  
  # Calculate the correlation between sample and group timeseries
  corr = cor(sample, group, method = 'spearman')
  
  return (corr)
}
