
##----------------------------------------------------------------------------##
## R script for Regression vs Machine Learning Paper
##----------------------------------------------------------------------------##


rm(list=ls())  
library(rms)
library(caret)
library(SuperLearner)
library(parallel)
library(doParallel)
library(dplyr)


physiorvml<- readRDS("physiorvml.Rds") 

iter <- 200 ## create 200 repeated random splits (outer cross-validation)
W <- NULL   ## to store the model performance metrics 
