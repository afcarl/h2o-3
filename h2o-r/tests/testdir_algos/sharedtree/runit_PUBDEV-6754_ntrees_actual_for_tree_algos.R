setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")



test.tree_algos.bernoulli <- function() {
  Log.info("Importing prostate.csv data...\n")
  prostate.hex <- h2o.uploadFile(locate("smalldata/logreg/prostate.csv"), destination_frame="prostate.hex")
  Log.info("Converting CAPSULE and RACE columns to factors...\n")
  prostate.hex$CAPSULE <- as.factor(prostate.hex$CAPSULE)
  prostate.hex$RACE <- as.factor(prostate.hex$RACE)
  Log.info("Summary of prostate.csv from H2O:\n")
  print(summary(prostate.hex))
  
  # Import csv data for R to use in comparison
  prostate.data <- read.csv(locate("smalldata/logreg/prostate.csv"), header = TRUE)
  prostate.data$RACE <- as.factor(prostate.data$RACE)
  Log.info("Summary of prostate.csv from R:\n")
  print(summary(prostate.data))
  
  # Train H2O GBM Model:
  ntrees <- 1000
  Log.info(paste("H2O GBM with parameters:\nnfolds = 5, distribution = 'bernoulli', ntrees = ", ntrees, ", stopping_metric=\"MSE\", stopping_tolerance=0.01, stopping_rounds=5\n", sep = ""))
  prostate_gbm.h2o <- h2o.gbm(x = 3:9, y = "CAPSULE", training_frame = prostate.hex, nfolds = 5, distribution = "bernoulli", ntrees = ntrees, stopping_metric="MSE", stopping_tolerance=0.01, stopping_rounds=5)
 
  Log.info("GBM Model: number of trees set by user before building the model is:"); print(ntrees)
  Log.info("GBM Model: number of trees built with early-stopping is:"); print(h2o.get_ntrees_actual(prostate_gbm))

  expect_true(h2o.get_ntrees_actual(prostate_gbm) < ntrees)
  expect_equal(h2o.get_ntrees_actual(prostate_gbm), prostate_gbm.h2o@model$model_summary['number_of_trees'][,1])

  # Train H2O Isolation Forest Model:
  Log.info(paste("H2O Isolation Forest with parameters:\nsample_rate = 0.1, max_depth = 20, ntrees = ", ntrees, ", stopping_metric=\"anomalyscore\", stopping_tolerance=0.01, stopping_rounds=5\n", sep = ""))
  prostate_if.h2o <- h2o.isolationForest(sample_rate = 0.1, max_depth = 20, ntrees=ntrees, stopping_metric="anomalyscore", stopping_tolerance=0.01, stopping_rounds=5)

  Log.info("Isolation Forest Model: number of trees set by user before building the model is:"); print(ntrees)
  Log.info("Isolation Forest Model: number of trees built with early-stopping is:"); print(h2o.get_ntrees_actual(prostate_if))

  expect_true(h2o.get_ntrees_actual(prostate_if) < ntrees)
  expect_equal(h2o.get_ntrees_actual(prostate_if), prostate_if.h2o@model$model_summary['number_of_trees'][,1]) 

  # Train H2O Random Forest Model:
  Log.info(paste("H2O Random Forest with parameters:\nmax_depth=20, min_rows=10, ntrees = ", ntrees, ", stopping_metric=\"auc\", stopping_tolerance=0.01, stopping_rounds=5\n", sep = ""))
  prostate_rf.h2o <- h2o.randomForest(ntrees=ntrees, max_depth=20, min_rows=10, stopping_metric="auc", stopping_tolerance=0.01, stopping_rounds=5)

  Log.info("Random Forest Model: number of trees set by user before building the model is:"); print(ntrees)
  Log.info("Random Forest Model: number of trees built with early-stopping is:"); print(h2o.get_ntrees_actual(prostate_rf))

  expect_true(h2o.get_ntrees_actual(prostate_rf) < ntrees)
  expect_equal(h2o.get_ntrees_actual(prostate_rf), prostate_rf.h2o@model$model_summary['number_of_trees'][,1])

  # Train H2O XGBoost Model:
  Log.info(paste("H2O XGBoost with parameters:\ndistribution=\"auto\", seed=1, ntrees = ", ntrees, ", stopping_metric=\"auc\", stopping_tolerance=0.01, stopping_rounds=5\n", sep = ""))
  prostate_xgb <- h2o.xgboost(distribution="auto", ntrees=ntrees, seed=1, stopping_metric="auc", stopping_tolerance=0.01, stopping_rounds=5)

  Log.info("XGBoost Model: number of trees set by user before building the model is:"); print(ntrees)
  Log.info("XGBoost Model: number of trees built with early-stopping is:"); print(h2o.get_ntrees_actual(prostate_xgb))

  expect_true(h2o.get_ntrees_actual(prostate_xgb) < ntrees)
  expect_equal(h2o.get_ntrees_actual(prostate_xgb), prostate_xgb.h2o@model$model_summary['number_of_trees'][,1])  

}

doTest("GBM Test: provide actual ntree value", test.tree_algos.bernoulli)
