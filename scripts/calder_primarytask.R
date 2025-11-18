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

# Model 1: Multinomial Regression for Multiclass Label

library(glmnet)
# fit enet model
#alpha_enet <- 0.2
#fit_reg_multi <- glmnet(x = claims_training_input, 
                      # y = claims_training_label_multi, 
                        #family = 'multinomial',
                       # alpha = alpha_enet)

 #choose a strength by cross-validation
#set.seed(2084)
#cvout_multi <- cv.glmnet(x = claims_training_input, 
                         #y = claims_training_label_multi, 
                         #family = 'multinomial',
                         #alpha = alpha_enet)
# view results
#cvout
# glmnet produces a matrix error where it says x should be a matrix with 2 or
# more columns

# Model 2: Model with Relu Activation function for Binary Classification
set.seed(2084)
model <- keras_model_sequential() %>%
  preprocess_layer %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_dropout(0.1) %>% 
  layer_activation(activation = 'relu') %>% 
  layer_activation(activation = 'sigmoid')

summary(model)

# configure for training
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)



# train
history <- model %>%
  fit(claims_training_input, 
      claims_training_label_binary,
      validation_split = 0.2,
      epochs = 15)

plot(history)

keras3::get_weights(model)

evaluate(model, claims_training_input, claims_training_label_binary)

save_model(model, "results/validation_model.keras")

#load("data/claims-test.RData")

#source('scripts/prelim_2_Wendy/Prelim_2_Preprocessing.R')

#test_clean = claims_test %>% 
 # parse_data()

# parse_data() keeps giving an error in read_xml.raw()

#test_result = model %>% predict(claims_test)


