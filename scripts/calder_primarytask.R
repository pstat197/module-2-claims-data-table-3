# Need to do primary task and deliverable 3
library(tidyverse)
library(tidymodels)
library(tensorflow)
library(keras3)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tokenizers)

# Loading in the cleaned data
load("data/claims-clean-example.RData")

# need the following vars by the end: id, bclass.pred (binary predicted label),
# and mclass.pred - label for multiclass

# Building the Split
set.seed(1984)
claims_partition = claims_clean %>% 
  select(c(.id, label, mclass, bclass, text_clean)) %>% 
  initial_split(0.8)

# Creating the training input in the clean text and the training output in the
# bclass and mclass

claims_training_input <- training(claims_partition) %>%
  pull(text_clean)
claims_training_label_binary <- training(claims_partition) %>%
  pull(bclass) %>%
  as.numeric() - 1
claims_training_label_multi <- training(claims_partition) %>%
  pull(mclass)

# create a preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)

# Preprocess the clean text
preprocess_layer %>% adapt(claims_training_input)

# Binary classification

set.seed(2084)
model_binary <- keras_model_sequential() %>%
  preprocess_layer() %>% 
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")  # binary output

model_binary %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

binary_history <- model_binary %>%
 fit(claims_training_input, 
  claims_training_label_binary,
  validation_split = 0.2,
  epochs = 15)

summary(model_binary)

set.seed(2087)
# Multiclass classification (5 classes)
model_multi <- keras_model_sequential() %>%
  preprocess_layer() %>% 
  #layer_input(shape = 1) %>% 
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 5, activation = "softmax")  # 5-class output

model_multi %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",  # For one-hot encoded labels
  metrics = c("accuracy")
)

# Encode to One-Hot Encoding to match categorical cross-entropy
claims_training_label_multi = to_categorical(claims_training_label_multi, 
                                             num_classes = 5)
  
multi_history <- model_multi %>%
  fit(claims_training_input, 
      claims_training_label_multi,
      validation_split = 0.2,
      epochs = 15)

summary(model_multi)

# Quick Stat for Binary Class Model

evaluate(model_binary, claims_training_input, claims_training_label_binary)

# Quick stat for Multiclass Model

evaluate(model_multi, claims_training_input, claims_training_label_multi)

# Predicting on Test Data

load("data/claims-test.RData")

source('scripts/LorrettaScript.R')

# preprocess (will take a minute or two)
claims_test_clean <- claims_test %>%
  parse_data2()


#binary_predict = model_binary %>% predict(claims_test$text_tmp)


