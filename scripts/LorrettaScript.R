# Lorretta Lu
# ============================================================================
# Preliminary Task 1
# Augment the HTML scraping strategy so that header information is captured in
# addition to paragraph content. Are binary class predictions improved using
# logistic principal component regression?
# ============================================================================
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)
library(tidyverse)
library(tidymodels)
library(textrecipes)

# original function to parse html and clean text
# parse_fn <- function(.html){
#   read_html(.html) %>%
#     html_elements('p') %>%
#     html_text2() %>%
#     str_c(collapse = ' ') %>%
#     rm_url() %>%
#     rm_email() %>%
#     str_remove_all('\'') %>%
#     str_replace_all(paste(c('\n', 
#                             '[[:punct:]]', 
#                             'nbsp', 
#                             '[[:digit:]]', 
#                             '[[:symbol:]]'),
#                           collapse = '|'), ' ') %>%
#     str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
#     tolower() %>%
#     str_replace_all("\\s+", " ")
# }

# attempting to augment original function
augment_parse_fn <- function(.html) {
  doc <- read_html(.html)
  
  # extract paragraph text
  para_text <- doc %>% 
    html_elements('p') %>% 
    html_text2() %>% 
    str_c(collapse = ' ')
  
  # extract headers
  header_text <- doc %>% 
    html_elements('h1, h2, h3, h4, h5, h6') %>% 
    html_text2() %>% 
    str_c(collapse = ' ')
  
  # combine headers and paragraphs
  full_text <- str_c(header_text, para_text, sep = ' ')
  
  # apply existing cleaning pipeline
  full_text %>% 
    rm_url() %>% 
    rm_email() %>% 
    str_remove_all('\'') %>% 
    str_replace_all(paste(c('\n',
                            '[[:punct:]]',
                            'nbsp',
                            '[[:digit:]]',
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>% 
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>% 
    tolower() %>% 
    str_replace_all("\\s+", " ")
}

# load claims-raw.RData
load('data/claims-raw.RData')

# preprocessing.R functions
# function to apply to claims data
parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

# next step: save claims-clean.RData
source('scripts/preprocessing.R')
load('data/claims-raw.RData')
claims_clean <- claims_raw %>% 
  parse_data()
save(claims_clean, file = 'data/claims-clean.RData')

# logistic PCR ----------------------------------------------------------------
source('scripts/preprocessing.R')

# train/test split
set.seed(42)
split <- initial_split(claims_clean, prop = 0.8, strata = bclass)
train <- training(split)
test <- testing(split)

# build the recipe
pcr_rec <- recipe(bclass ~ text_clean, data = train) %>% 
  step_tokenize(text_clean) %>% 
  step_stopwords(text_clean) %>% 
  step_tf(text_clean) %>% 
  step_pca(all_predictors(), num_comp = 50)

# specify logistic regression model
logit_mod <- logistic_reg(mode = "classification") %>% 
  set_engine("glm")

# build and fit workflow
pcr_wf <- workflow() %>% 
  add_recipe(pcr_rec) %>% 
  add_model(logit_mod)
pcr_fit <- pcr_wf %>% 
  fit(data = train)

# predict and evaluate performance
pcr_preds <- predict(pcr_fit, test, type = "prob") %>% 
  bind_cols(predict(pcr_fit, test)) %>% 
  bind_cols(test %>% select(bclass))

# define metrics accuracy: 0.723
metrics_binary <- metric_set(accuracy)
metrics_binary(
  data = pcr_preds,
  truth = bclass,
  estimate = .pred_class,
  event_level = "second"
)

# define metrics roc auc: 0.760
roc_auc(
  data = pcr_preds,
  truth = bclass,
  `.pred_Relevant claim content`,
  event_level = "second"
)

# =============================================================================
# An accuracy of 0.723 means the model correctly classified about 72.3% of
# webpages in the test set
# An ROC AUC of 0.760 means the model is better at random guessing and can
# separate the two classes reasonably well
# =============================================================================

# loading baseline model ------------------------------------------------------
# load cleaned data
load('data/claims-clean-example.RData')

# partition
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

train_df <- training(partitions)
test_df  <- testing(partitions)

pcr_rec_text <- recipe(bclass ~ text_clean, data = train_df) %>%
  step_tokenize(text_clean) %>%
  step_tfidf(text_clean) %>%
  step_pca(starts_with("tfidf_"), num_comp = 50)

pcr_mod_text <- logistic_reg() %>%
  set_engine("glm")

pcr_wf_text <- workflow() %>%
  add_recipe(pcr_rec) %>%
  add_model(pcr_mod_text)

pcr_fit_text <- fit(pcr_wf_text, data = train_df)

pcr_preds_text <- predict(pcr_fit_text, test_df, type = "prob") %>%
  bind_cols(predict(pcr_fit, test_df, type = "class")) %>%
  bind_cols(test_df %>% select(bclass))

metrics_text <- metric_set(accuracy, roc_auc)

metrics_text(
  data    = pcr_preds_text,
  truth   = bclass,
  estimate = .pred_class,
  `.pred_N/A: No relevant content.`,
  event_level = "second"
)

# Baseline Accuracy: 0.766
# ROC AUC Baseline: 0.208

# =============================================================================
# Preliminary Task 1 Question
# Are binary class predictions improved using logistic principal component
# regression?
# No, binary class predictions were not improved in terms of classification
# accuracy using logistic principal component regression (as the accuracy was
# decreased from 0.766 to 0.723). However, logistic PCR provided a substantial 
# improvement in discriminatory ability, increasing ROC AUC from 0.208 to 0.760,
# indicating that although accuracy decreased slightly, the PCR model produced
# far more meaningful probability estimates and separated the classes much
# better. Overall, if accuracy is the metric of interest, the baseline model
# performed better. If ranking ability and probability quality matter, logistic
# PCR is preferred.
# =============================================================================