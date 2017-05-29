setwd("C:/Users/joerj/OneDrive/Documents/cs231n/project")
library(readr)
library(text2vec)
library(tokenizers)
library(tm)
library(ggplot2)

# Take training labels and generate matrix indicating their prevalence
data <- read_csv("train.csv")

tag_words <- tokenize_ngrams(data$tags, n = 1, n_min = 1)

tokens <- itoken(tag_words, 
                 preprocessor = words, 
                 tokenizer = word_tokenizer, 
                 progressbar = FALSE)

vocab = create_vocabulary(tokens)
vectorizer <- vocab_vectorizer(vocab, grow_dtm = T)

tags_matrix <- as.matrix(create_dtm(tokens, vectorizer))

total_labels <- nrow(tags_matrix)

tags_dist <- colSums(tags_matrix)
names(tags_dist) <- gsub("_", " ", names(tags_dist))

bar_chart_data <- data.frame(names(tags_dist), tags_dist / total_labels * 100)
rownames(bar_chart_data) <- NULL
colnames(bar_chart_data) <- c("Label", "Percent of Training Sample")
# Bar 
pdf("label_distribution.pdf", width = , height = )
ggplot(data = bar_chart_data, aes(x = Label, y = `Percent of Training Sample`)) + 
  geom_bar(stat='identity', fill = "steelblue3")  + 
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
dev.off()