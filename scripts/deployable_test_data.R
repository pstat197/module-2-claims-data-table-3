library(tidyverse)
library(keras3)
library(here)

## Deliverable 4

load(here("data/claims-test-clean.RData"))
claims_testing <- claim_clean_testing %>% pull(text_clean)


# loading models - This code does not execute as Keras can't deserialize the
# text vectorization/stringlookup layers in the current environment

# The dataframe provided in results shows that the model did work successfully
# on the test data
binary_model <- load_model(here("results/binary-model.keras"))
multi_model  <- load_model(here("results/multi-model.keras"))

# predictions
binary_pred <- predict(binary_model, claims_testing)
multi_pred  <- predict(multi_model, claims_testing)

# results
data.frame(ID = claim_clean_testing$.id, Prob = binary_pred[,1]) %>% head(10)
data.frame(ID = claim_clean_testing$.id, multi_pred) %>% head(10)