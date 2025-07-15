library(readr)
library(dplyr)
library(caret)
library(e1071)
library(nnet)
library(pROC)
library(ggplot2)
library(viridis)

data2010 <- read_csv("C:/Users/dell/Desktop/date/RO_LFS_2010_Y.csv")
data2011 <- read_csv("C:/Users/dell/Desktop/date/RO_LFS_2011_Y.csv")
data2012 <- read_csv("C:/Users/dell/Desktop/date/RO_LFS_2012_Y.csv")
data2013 <- read_csv("C:/Users/dell/Desktop/date/RO_LFS_2013_Y.csv")

data <- bind_rows(data2010, data2011, data2012, data2013)
data$unemployed <- ifelse(data$ILOSTAT == 2, 1, 0)
data$unemployed <- as.factor(data$unemployed)

data <- data %>%
  select(unemployed, AGE, SEX, HATLEV1D, REGION, STAPRO, NACE1D, ISCO1D)

set.seed(123)
trainIndex <- createDataPartition(data$unemployed, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]
write_csv(train, "C:/Users/dell/Desktop/date/train.csv")
write_csv(test, "C:/Users/dell/Desktop/date/test.csv")

categorical_cols <- c("SEX", "HATLEV1D", "REGION", "STAPRO", "NACE1D", "ISCO1D")
train[categorical_cols] <- lapply(train[categorical_cols], as.factor)
test[categorical_cols] <- lapply(test[categorical_cols], as.factor)
train$unemployed <- as.factor(train$unemployed)
test$unemployed <- as.factor(test$unemployed)

remove_single_level <- function(df) {
  df[, sapply(df, function(col) length(unique(col)) > 1)]
}
train <- remove_single_level(train)
test <- remove_single_level(test)

if (length(unique(train$unemployed)) < 2) {
  stop("Eroare: doar o clasă în train$unemployed.")
}

set.seed(123)
sample_train_svm <- train[sample(nrow(train), min(5000, nrow(train))), ]
test_svm <- test[complete.cases(test), ]

model_svm <- svm(unemployed ~ ., data = sample_train_svm, kernel = "radial", probability = TRUE)
pred_svm <- predict(model_svm, test_svm)
conf_svm <- confusionMatrix(factor(pred_svm, levels = levels(test_svm$unemployed)), test_svm$unemployed)

pred_svm_probs <- attr(predict(model_svm, test_svm, probability = TRUE), "probabilities")[,2]
roc_svm <- roc(as.numeric(test_svm$unemployed), pred_svm_probs)

set.seed(123)
sample_train_mlp <- train[sample(nrow(train), min(5000, nrow(train))), ]
sample_test_mlp <- test[sample(nrow(test), min(2000, nrow(test))), ]

preproc <- preProcess(sample_train_mlp, method = c("center", "scale"))
train_mlp <- predict(preproc, sample_train_mlp)
test_mlp <- predict(preproc, sample_test_mlp)

train_mlp$unemployed <- sample_train_mlp$unemployed
test_mlp$unemployed <- sample_test_mlp$unemployed

model_mlp <- nnet(unemployed ~ ., data = train_mlp, size = 4, maxit = 100, linout = FALSE, trace = FALSE)

pred_mlp_raw <- predict(model_mlp, test_mlp, type = "raw")
pred_mlp <- ifelse(pred_mlp_raw > 0.5, 1, 0)
pred_mlp <- factor(pred_mlp, levels = c(0, 1))
conf_mlp <- confusionMatrix(pred_mlp, test_mlp$unemployed)

roc_mlp <- roc(as.numeric(test_mlp$unemployed), pred_mlp_raw)

png("C:/Users/dell/Desktop/date/curba_roc.png")
plot(roc_svm, col = "blue", main = "Curba ROC")
lines(roc_mlp, col = "red")
legend("bottomright", legend = c(
  paste0("SVM (AUC = ", round(auc(roc_svm), 2), ")"),
  paste0("MLP (AUC = ", round(auc(roc_mlp), 2), ")")
), col = c("blue", "red"), lwd = 2)
dev.off()

cat("=== MATRICE DE CONFUZIE: SVM ===\n")
print(conf_svm)

cat("\n=== MATRICE DE CONFUZIE: MLP ===\n")
print(conf_mlp)

cat("\nAUC SVM: ", auc(roc_svm), "\n")
cat("AUC MLP: ", auc(roc_mlp), "\n")

# === Salvare Matrice Confuzie SVM ===
svm_table <- as.table(conf_svm$table)
svm_df <- as.data.frame(svm_table)
colnames(svm_df) <- c("Prediction", "Reference", "Freq")

ggplot(data = svm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_viridis_c() +
  labs(title = "Matrice Confuzie – SVM") +
  theme_minimal()

ggsave("C:/Users/dell/Desktop/date/matrice_svm.png", width = 5, height = 5)

# === Salvare Matrice Confuzie MLP ===
mlp_table <- as.table(conf_mlp$table)
mlp_df <- as.data.frame(mlp_table)
colnames(mlp_df) <- c("Prediction", "Reference", "Freq")

ggplot(data = mlp_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_viridis_c() +
  labs(title = "Matrice Confuzie – MLP") +
  theme_minimal()

ggsave("C:/Users/dell/Desktop/date/matrice_mlp.png", width = 5, height = 5)