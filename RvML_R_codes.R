
##----------------------------------------------------##
## R script for Regression vs Machine Learning Paper
##----------------------------------------------------##


rm(list=ls())  
library(rms)
library(caret)
library(SuperLearner)
library(parallel)
library(doParallel)
library(dplyr)
library(vip)  


physiorvml<- readRDS("physiorvml.Rds") 

iter <- 200 ## create 200 repeated random splits (outer cross-validation)
W <- NULL   ## to store the model performance metrics 

## speed up computation
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)


for (i in 1:iter) {
  
  set.seed(i)
  
  # Partitioning dataset (70-30 split)
  inTrain <- createDataPartition(physiorvml$opyear, p = 0.70, list = FALSE)
  
  ## Train Dataframe  
  data <- physiorvml[inTrain,]                            ## train dataframe 
  dx <- data[,-c( 1: 5 )]  				    	                  ## train predictors
  dy <-   as.factor(1*(as.numeric(data$walk4.24) >=3)) 		## train (binary) outcome
  levels(dy) <- c("none", "difficulty")  			            ## give level names else train.caret would balk
  dy.ordinal <- ordered(data$walk4.24)                   	## train ordinal outcome
  trainrms  <- data.frame(dx, dy.ordinal)                 ## train dataframe for rms
  trainrms  <- trainrms %>%  dplyr::select(-none, -chinese) ## to fulfill fullrank assumptions
  dx.sl <- trainrms %>%  dplyr::select(-dy.ordinal)
  dy.sl <- as.numeric(dy) - 1
  
  
  ## Test Dataframe  
  data.test <- physiorvml[-inTrain,]                          ## test dataframe 
  dx2 <- data.test[,-c( 1:5 )]        			              	  ## test predictors
  dy2 <-   1*(as.numeric(data.test$walk4.24) >=3)	 	          ## test (binary) outcome
  dx2.sl <- dx2 %>%  dplyr::select(-none, -chinese)                   
  
  
  ## set up trainControl() arguments for train.caret()
  myctrl1 <- trainControl(method="repeatedcv", number=5, repeats = 3, verboseIter=TRUE,
                          returnData=FALSE,
                          classProbs = TRUE,
                          ## summaryFunction = twoClassSummary
                          summaryFunction =  mnLogLoss
  ) 
  
  metric="logLoss"   
  

  
  ##----------------------------------##
  ## Logistc Regression: Binary & Ordinal
  ##----------------------------------##
  
  modlrm1 <-  lrm (as.numeric(dy.ordinal) >= 3 ~. , data = trainrms,  x = T, y = T)   ## logistic regression (assumes linearity) 
  modorm1 <- lrm (dy.ordinal ~. , data = trainrms,  x = T, y = T)
  modorm<- update(modorm1, .~.
                  - (age + weight + height + oactual.24 + flex.0 + ext.0 + sfpf.0 )
                  + (rcs(age,3) + rcs(weight,3) + rcs(height,3) + rcs(oactual.24,3) + rcs(flex.0,3)  + rcs(ext.0,3) + rcs(sfpf.0,3) )   )   ## orm with natural splines


  ##---------------------------------------##
  ## Logistic + LASSO/RIDGE/ELASTIC NET 
  ##---------------------------------------##

  
  ## LASSO (L1 penalty norm)
  modlasso <- train(dx, dy,  
                    method = "glmnet", 
                    trControl = myctrl1, 
                    metric = metric,
                    preProcess = c('scale','center'),
                    tuneGrid = expand.grid(alpha = 0,lambda =10^seq(0.5, -2, length = 400))
  )
  
  ## RIDGE (L2 penalty norm)
  modridge <- train(dx , dy , 
                    method = "glmnet", 
                    trControl = myctrl1, 
                    metric = metric,
                    preProcess = c('scale','center'),
                    tuneGrid = expand.grid(alpha = 0,lambda =10^seq(0.5, -2, length = 400))    
  )
  
  
  ## ELASTIC NET (Fitting alpha = 0.0769, lambda = 0.0238 on full training set)
  modenet <- train(dx , dy , 
                   method = "glmnet", 
                   trControl = myctrl1, 
                   metric = metric,
                   preProcess = c('scale','center'),
                   tuneGrid = expand.grid(alpha = seq(0, 1, length = 40) , lambda =10^seq(0.5, -2, length = 400))   
  )
  
  
  
  ##--------------##
  ## Random Forest
  ##--------------##
    
  mtry1 <- floor(sqrt(ncol(dx)) * c(0.2, 0.5, 0.7, 1, 1.5, 2, 2.2, 2.4)) ##  1,  2,  3,  5,  8, 10, 12, 13
  
  ## Random Forest
  modrf <-   train(dx , dy , 
                   method = "rf", 
                   trControl = myctrl1, 
                   metric = metric,
                   importance=T, 
                   tuneGrid=expand.grid( mtry=mtry1 ) , ntree=1000)
  
    
    
    ##------------##
    ## XGBoost
    ##------------##

    modxgbm  <-   train(dx , dy , 
                        method = "xgbTree", 
                        trControl = myctrl1, 
                        metric = metric,
                        tuneGrid= expand.grid(
                          eta = c(0.01, 0.025, 0.050, 0.075, 0.10, 0.20, 0.30), 
                          nrounds = c(500, 1000, 1500, 2000),  
                          max_depth = 2:6,   
                          min_child_weight = 1:6 , 
                          gamma = c(0,6),    
                          colsample_bytree = seq(0.3, 1.0, 0.1),  
                          subsample = seq(0.5, 1, length = 5) )    )
   
    
    ##--------------------##
    # SuperLearner
    ##--------------------##
    
    ##------------------ Step 1: create Learners------------------------## 
    
    ## Use tuned hyperparameters from caret analyses
    myval <- modxgbm$bestTune
    SL.xgboost1 <- function(...){   SL.xgboost(..., 
                                               ntrees = myval$nrounds, 
                                               max_depth = myval$max_depth, 
                                               shrinkage = myval$eta,    
                                               minobspernode =  myval$min_child_weight,
                                               params = list(
                                               gamma = myval$gamma, 
                                               colsample_bytree = myval$colsample_bytree,
                                               subsample = myval$subsample)) 
    }
    
    myval1 <- modrf$bestTune
    SL.randomForest1 <- function(...){   SL.randomForest(..., tune = myval1$mtry)    
      }

    learners1 <- create.Learner("SL.glmnet", detailed_names=T, tune=list(alpha = 0) )   ## Ridge (L2 norm) regression 

    
    ##------------------ Step 2: Run the SuperLearner------------------------## 

    mylibrarysnow  <- c( 
      "SL.glmnet",
      "SL.glm",
      "SL.xgboost1",  
      "SL.randomForest1",  
      learners1$names)
    
    # Make a snow cluster 
    cluster = parallel::makeCluster(detectCores()-1); cluster
    
    # Load the SuperLearner package on all workers so they can find
    parallel::clusterEvalQ(cluster, library(SuperLearner))
    
    # Explictly export custom learner functions to the workers
    parallel::clusterExport(cluster, c(learners1$names, "SL.xgboost1","SL.randomForest1", "myval", "myval1"))
    
    # We need to set a different type of seed that works across cores.
    parallel::clusterSetRNGStream(cluster, 1)
    
    modslsnow <- snowSuperLearner(Y = dy.sl, X = dx.sl,  
                                    family="binomial",
                                    method='method.NNloglik' , 
                                    cvControl=list(V=5), 
                                    cluster = cluster, 
                                    verbose = TRUE,
                                    SL.library = mylibrarysnow) 

    ##---------------------------------##
    # Collate performance metrics
    ##---------------------------------##
    
    
    modlist <- list (logistic = modlrm1,
                     ordinal = modorm, 
                     lasso = modlasso, 
                     ridge = modridge, 
                     elasticnet = modenet, 
                     randomforest = modrf,
                     extreme_gbm = modxgbm,
                     superlearner = modslsnow)
    
    
    U <- NULL
    for (j in names (modlist)) {
      if (j == "logistic")              {p <- plogis (predict(modlrm1, data.test))
      } else if (j == "ordinal")        {p <- plogis (predict(modorm, data.test, kint = 2)) 
      } else if (j == "superlearner")   {p <- predict (modslsnow, newdata=dx2.sl, onlySL = T)$pred
      } else                            {p <- predict (modlist[[j]], dx2 , type = "prob")[[2]] }  
      
      
      
      
      
      aa <- data.frame (p = p , y = dy2) %>%  dplyr::filter( p <= 1 ) 
      
      ## Harrell's val.prob.R generates AUC, eavg (MAE), and Brier score
      mystats <- with(aa, val.prob(p = p, y = y, logistic.cal=FALSE, pl=FALSE))
      
      u <- data.frame(
        model = j, 
        cindex = mystats[2], 
        eavg = mystats[16],  
        brier = mystats[11])
      
      U <- rbind(U, u)
      rownames(U) <- NULL
      
    }  # end for-loop to consolidate the model statistics for one iteration
    
    W <- rbind (W, U)  ## consolidate model statistics over 200 iterations
    
} ## end nested cross-validation
  


saveRDS(W, file="rvml.Rds")  



