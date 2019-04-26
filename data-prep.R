library(mlflow)
library(sparktf)
library(sparklyr)

sc <- spark_connect(master = "local")

# Create temporary directories to write data
path_train_data <- file.path(tempdir(), "iris_train")
path_validation_data <- file.path(tempdir(), "iris_validation")

# Pretend this is a massive distributed dataset...
iris_tbl <- copy_to(sc, iris)

# Log some info
mlflow_log_param("dataset", "iris")

# Create a Spark ML pipeline
pipeline <- ml_pipeline(sc) %>% 
  ft_string_indexer_model(
    "Species", "label",
    labels = c("setosa", "versicolor", "virginica")
  ) %>% 
  ft_vector_assembler(
    c("Petal_Length", "Petal_Width", "Sepal_Length", "Sepal_Width"), 
    "features") %>% 
  ft_standard_scaler("features", "features_scaled", with_mean = TRUE)

# Train/Validation split
splits <- sdf_random_split(iris_tbl, training = 0.8, validation = 0.2)

# Fit the pipeline model
pipeline_model <- pipeline %>% 
  ml_fit(splits$training)

# Use model to transform training and validation data
pipeline_model %>% 
  ml_transform(splits$training) %>% 
  spark_write_tfrecord(
    path = path_train_data,
    write_locality = "local"
  )

pipeline_model %>% 
  ml_transform(splits$validation) %>% 
  spark_write_tfrecord(
    path = path_validation_data,
    write_locality = "local"
  )

# Log processed datasets
mlflow_log_artifact(path_train_data, artifact_path = "iris_train")
mlflow_log_artifact(path_validation_data, artifact_path = "iris_validation")

mlflow_end_run()
