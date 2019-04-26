library(mlflow)
library(keras)
library(tensorflow)
library(tfdatasets)

# Declare parameters
path_train_data <- mlflow_param("path_train_data", type = "string")
path_validation_data <- mlflow_param("path_validation_data", type = "string")
num_units <- mlflow_param("num_units", 32)
epochs <- mlflow_param("epochs", 100)
learning_rate <- mlflow_param("learning_rate", 1e-3)

# Function to parse TFRecord file
parse_dataset <- function(data_path) {
  tfrecord_dataset(list.files(data_path, full.names = TRUE)) %>%
    dataset_map(function(example_proto) {
      features <- list(
        features_scaled = tf$FixedLenFeature(shape(4), tf$float32),
        label = tf$FixedLenFeature(shape(), tf$float32)
      )
      
      features <- tf$parse_single_example(example_proto, features)
      x <- features$features_scaled
      y <- tf$one_hot(tf$cast(features$label, tf$int32), 3L)
      list(x, y)
    }) %>%
    dataset_shuffle(256) %>%
    dataset_batch(16) %>% 
    dataset_repeat()
}

dataset_train <- parse_dataset(path_train_data)
dataset_validation <- parse_dataset(path_validation_data)

# iter <- make_iterator_one_shot(dataset_train)
# iterator_get_next(iter)

# Build fancy model
model <- keras_model_sequential() %>%
  layer_dense(num_units, input_shape = 4, activation = "relu") %>% 
  layer_dense(3, activation = "softmax")

model %>% compile(
  optimizer = optimizer_adam(lr = learning_rate),
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

# Train model
history <- model %>% 
  fit(dataset_train, epochs = epochs, steps_per_epoch = 10,
      verbose = 0)

# Save training plot
p <- plot(history)
ggplot2::ggsave("training_history.png", plot = p)
mlflow_log_artifact("training_history.png")

# Log metrics and model
metrics <- model %>% evaluate(dataset_validation, steps = 1)
mlflow_log_metric("accuracy", metrics$acc)
mlflow_log_model(model, "keras_model")
mlflow_end_run()
