library(mlflow)
library(dplyr)

mlflow_run("data-prep.R")

run_infos <- mlflow_list_run_infos(experiment_id = "0")
run_infos
data_prep_run_id <- run_infos %>% 
  filter(entry_point_name == "data-prep.R",
         status == "FINISHED") %>% 
  pull(run_uuid)

path_train_data <- mlflow_download_artifacts("iris_train", run_id = data_prep_run_id)
path_validation_data <- mlflow_download_artifacts("iris_validation", run_id = data_prep_run_id)

mlflow_run("train-model.R", param_list = list(
  path_train_data = path_train_data,
  path_validation_data = path_validation_data,
  num_units = 16,
  epochs = 10,
  learning_rate = 0.005
))

run_infos <- mlflow_list_run_infos(experiment_id = "0")
training_run_id <- run_infos %>% 
  filter(entry_point_name == "train-model.R",
         status == "FINISHED") %>% 
  pull(run_uuid)

mlflow_rfunc_serve(
  "keras_model", 
  run_uuid = training_run_id
)

