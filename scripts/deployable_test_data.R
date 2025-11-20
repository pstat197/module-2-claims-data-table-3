library(tidyverse)
library(keras3)
library(here)

load(here("data/claims-test-clean.RData"))
claims_testing <- claim_clean_testing %>% pull(text_clean)

# loading models
binary_model <- load_model(here("results/binary-model.keras"))
multi_model  <- load_model(here("results/multi-model.keras"))

# predictions
binary_pred <- predict(binary_model, claims_testing)
multi_pred  <- predict(multi_model, claims_testing)

# results
data.frame(ID = claim_clean_testing$.id, Prob = binary_pred[,1]) %>% head(10)
data.frame(ID = claim_clean_testing$.id, multi_pred) %>% head(10)