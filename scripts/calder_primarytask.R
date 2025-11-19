# Need to do primary task and deliverable 3
library(tidyverse)
library(tidymodels)
library(tensorflow)
library(keras3)


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

load("data/claims-test-clean.RData")

# preprocess (will take a minute or two)
claim_testing <- claim_clean_testing %>% 
  pull(text_clean)

binary_predict = model_binary %>% predict(claim_testing)

multi_predict = model_multi %>% predict(claim_testing)

# Saving models and data frame

save_model(model_binary, filepath = "results/binary-model.keras")

save_model(model_multi, filepath = "results/multi-model.keras")

# Collapsing columns and setting up bclass and mclass columns

labels = c("Relevant Content", "N/A - No Relevant Content", "Physical Activity", 
           "Potentially Unlawful Activity", "Other Claim Content")

bclass_clean = c(1:915)
for(each_index in 1:915)
{
  if(binary_predict[each_index] >= 0.5)
  {
    bclass_clean[each_index] = labels[1]
  }
  else{
    bclass_clean[each_index] = labels[2]
  }
}


binary_df = data.frame(
  ID = claim_clean_testing$.id,
  bclass = bclass_clean
)

multi_df = data.frame(
  no_relevant_content = format(multi_predict[1:915,1], scientific = FALSE),
  physical_activity = format(multi_predict[1:915,2], scientific = FALSE),
  possible_fatality = format(multi_predict[1:915,3], scientific = FALSE),
  potentially_unlawful_activity = format(multi_predict[1:915,4], 
                                         scientific = FALSE),
  other_claim_content = format(multi_predict[1:915,5], scientific = FALSE)
)

collapsed_multi_df = data.frame(mclass = names(multi_df)[max.col(multi_df)])

preds_df = data.frame(binary_df, collapsed_multi_df)

# Saving models and data frame

# Deliverable 3 -  Store the data frame as an RData file. 

save(preds_df, file = "preds-group[3].RData")





