##############################################################################
## Ensemble Learning 作業：CART vs. Random Forest vs. XGBoost
##   Part A  Iris            (分類)  CART / RF / XGBoost
##   Part B  Boston Housing  (迴歸)  LM / CART / RF / XGBoost
##   Part C  Baseball        (迴歸)  LM / CART / RF / XGBoost，預測 Logsalary
##
## 使用方式：
##   1. 把這支程式和 ball.txt 放在同一個資料夾
##   2. 在 RStudio 開啟後按 Source (Ctrl + Shift + S)
##   3. 全部圖檔與數字會輸出到同資料夾下的 HW_output/
##############################################################################

## 若需要指定套件路徑，取消下一行註解並改成自己的路徑
# .libPaths("C:/Program Files/R/R-4.6.0/library")

#################### 0. 套件與共用設定 ####################
pkgs <- c("rpart", "rpart.plot", "randomForest", "xgboost", "caTools",
          "ggplot2", "GGally", "gridExtra", "mlbench")
new.pkgs <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(new.pkgs) > 0) install.packages(new.pkgs)
invisible(lapply(pkgs, library, character.only = TRUE))

## 工作目錄設為本程式所在資料夾 (RStudio)
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  src <- rstudioapi::getSourceEditorContext()$path
  if (nzchar(src)) setwd(dirname(src))
}
if (!file.exists("ball.txt")) stop("找不到 ball.txt，請把它和本程式放在同一個資料夾")

out_dir <- "HW_output"
dir.create(out_dir, showWarnings = FALSE)
fig <- function(name) file.path(out_dir, name)

## 結果紀錄檔
log_file <- fig("results_log.txt")
cat("Ensemble Learning HW results\n", format(Sys.time()), "\n", file = log_file)
logp <- function(title, x) {
  cat("\n=====", title, "=====\n", file = log_file, append = TRUE)
  capture.output(print(x), file = log_file, append = TRUE)
  cat("\n=====", title, "=====\n")
  print(x)
}

## 繪圖風格
theme_hw <- theme_bw(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())
col_model <- c("Linear Regression" = "#7F8C8D", "CART" = "#E67E22",
               "Random Forest" = "#1E8449", "XGBoost" = "#1F4E9C")
col_species <- c("setosa" = "#E67E22", "versicolor" = "#1E8449",
                 "virginica" = "#1F4E9C")

save_gg <- function(p, name, w = 7, h = 5) {
  ggsave(fig(name), p, width = w, height = h, dpi = 300)
}
save_base <- function(name, expr, w = 7, h = 5) {
  png(fig(name), width = w, height = h, units = "in", res = 300)
  on.exit(dev.off())
  expr
}

## 評估指標
metrics <- function(pred, obs) {
  mse <- mean((pred - obs)^2)
  c(MSE = mse, RMSE = sqrt(mse),
    R2 = 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2))
}

## 變數重要性長條圖
imp_bar <- function(v, title, xlab, fill, top = 10) {
  if (is.null(v) || length(v) == 0) {
    return(ggplot() + labs(title = paste(title, "(no split)")) + theme_void())
  }
  v <- sort(v, decreasing = TRUE)[seq_len(min(top, length(v)))]
  df <- data.frame(var = factor(names(v), levels = rev(names(v))),
                   value = as.numeric(v))
  ggplot(df, aes(x = value, y = var)) +
    geom_col(fill = unname(fill), width = 0.7) +
    labs(title = title, x = xlab, y = NULL) +
    theme_hw
}


#############################################################################
## Part A. Iris (Classification)
#############################################################################
data("iris")

## A-1 敘述統計圖
p.pairs <- ggpairs(iris, columns = 1:4,
                   aes(colour = Species, alpha = 0.7)) +
  scale_colour_manual(values = col_species) +
  scale_fill_manual(values = col_species) +
  theme_bw()
save_base("A1_iris_pairs.png", print(p.pairs), w = 9, h = 8)

box_one <- function(v) {
  ggplot(iris, aes(x = Species, y = .data[[v]], fill = Species)) +
    geom_boxplot() +
    scale_fill_manual(values = col_species) +
    labs(title = v, x = NULL, y = NULL) +
    theme_hw + theme(legend.position = "none")
}
save_gg(arrangeGrob(grobs = lapply(names(iris)[1:4], box_one), ncol = 2),
        "A2_iris_boxplots.png", w = 8, h = 7)

## A-2 訓練 / 測試資料 (與老師範例相同的切法)
set.seed(123)
split.iris <- sample.split(iris$Species, SplitRatio = 0.8)
train.iris <- subset(iris, split.iris == TRUE)
test.iris  <- subset(iris, split.iris == FALSE)
logp("Iris train / test dim", rbind(train = dim(train.iris), test = dim(test.iris)))

## A-3 CART
fit.iris <- rpart(Species ~ ., data = train.iris, method = "class")
save_base("A3_iris_cart_tree.png",
          rpart.plot(fit.iris, type = 2, extra = 104, roundint = FALSE,
                     main = "Iris: Classification Tree"),
          w = 8, h = 5.5)
pred.cart.iris <- predict(fit.iris, newdata = test.iris, type = "class")

## A-4 Random Forest
set.seed(123)
rf.iris <- randomForest(Species ~ ., data = train.iris,
                        ntree = 500, mtry = 2, importance = TRUE)
logp("Iris Random Forest", rf.iris)
pred.rf.iris <- predict(rf.iris, newdata = test.iris)

err <- as.data.frame(rf.iris$err.rate)
err.long <- data.frame(
  ntree = rep(seq_len(nrow(err)), times = ncol(err)),
  type  = factor(rep(colnames(err), each = nrow(err)), levels = colnames(err)),
  error = unlist(err, use.names = FALSE)
)
p.err <- ggplot(err.long, aes(ntree, error, colour = type)) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = c("OOB" = "black", col_species)) +
  labs(title = "Iris: Random Forest OOB Error",
       x = "Number of trees", y = "Error rate", colour = NULL) +
  theme_hw
save_gg(p.err, "A4_iris_rf_oob.png", w = 7, h = 4.5)

## A-5 XGBoost
x.train.iris <- as.matrix(train.iris[, 1:4])
x.test.iris  <- as.matrix(test.iris[, 1:4])
y.train.iris <- as.numeric(train.iris$Species) - 1
y.test.iris  <- as.numeric(test.iris$Species) - 1
dtrain.iris <- xgb.DMatrix(data = x.train.iris, label = y.train.iris)
dtest.iris  <- xgb.DMatrix(data = x.test.iris,  label = y.test.iris)

params.iris <- list(objective = "multi:softprob", num_class = 3,
                    eval_metric = "mlogloss", max_depth = 3, eta = 0.1,
                    subsample = 1, colsample_bytree = 1, tree_method = "hist")
set.seed(123)
xgb.iris <- xgb.train(params = params.iris, data = dtrain.iris,
                      nrounds = 100, verbose = 0)

prob.raw.iris <- predict(xgb.iris, newdata = dtest.iris)
if (is.matrix(prob.raw.iris)) {
  prob.xgb.iris <- prob.raw.iris
} else {
  prob.xgb.iris <- matrix(prob.raw.iris, ncol = 3, byrow = TRUE)
}
colnames(prob.xgb.iris) <- levels(iris$Species)
pred.xgb.iris <- factor(max.col(prob.xgb.iris) - 1, levels = 0:2,
                        labels = levels(iris$Species))

## A-6 準確率與混淆矩陣
iris.preds <- list("CART" = pred.cart.iris,
                   "Random Forest" = pred.rf.iris,
                   "XGBoost" = pred.xgb.iris)
iris.acc <- data.frame(Model = names(iris.preds),
                       Accuracy = sapply(iris.preds, function(p) mean(p == test.iris$Species)),
                       row.names = NULL)
logp("Iris test accuracy", iris.acc)
write.csv(iris.acc, fig("A_iris_accuracy.csv"), row.names = FALSE)

cm.list <- lapply(names(iris.preds), function(m) {
  tb <- as.data.frame(table(Predicted = iris.preds[[m]], Actual = test.iris$Species))
  tb$Model <- sprintf("%s (Acc = %.3f)", m, iris.acc$Accuracy[iris.acc$Model == m])
  logp(paste("Iris confusion matrix -", m),
       table(Predicted = iris.preds[[m]], Actual = test.iris$Species))
  tb
})
cm.all <- do.call(rbind, cm.list)
cm.all$Model <- factor(cm.all$Model, levels = unique(cm.all$Model))
p.cm <- ggplot(cm.all, aes(Actual, Predicted, fill = Freq)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "#F4F6F7", high = "#58D68D", guide = "none") +
  facet_wrap(~ Model) +
  labs(title = "Iris: Confusion Matrices (test set)") +
  theme_hw + theme(axis.text.x = element_text(angle = 30, hjust = 1))
save_gg(p.cm, "A5_iris_confusion.png", w = 11, h = 4.2)

## A-7 變數重要性
imp.xgb.iris <- as.data.frame(xgb.importance(model = xgb.iris))
g.imp.iris <- arrangeGrob(
  imp_bar(fit.iris$variable.importance, "CART", "Importance", col_model["CART"]),
  imp_bar(importance(rf.iris)[, "MeanDecreaseGini"], "Random Forest",
          "Mean decrease Gini", col_model["Random Forest"]),
  imp_bar(setNames(imp.xgb.iris$Gain, imp.xgb.iris$Feature), "XGBoost",
          "Gain", col_model["XGBoost"]),
  ncol = 3)
save_gg(g.imp.iris, "A6_iris_importance.png", w = 13, h = 3.8)
logp("Iris RF importance", importance(rf.iris))
logp("Iris XGBoost importance", imp.xgb.iris)


#############################################################################
## 迴歸共用流程：LM / CART(剪枝) / RF(調 mtry) / XGBoost(調 max_depth)
#############################################################################
run_regression <- function(train, test, yname, xvars, prefix, label, lims, seed = 123) {
  fml  <- reformulate(xvars, response = yname)
  y.tr <- train[[yname]]
  y.te <- test[[yname]]

  ## (1) Multiple linear regression
  lm.fit  <- glm(fml, family = gaussian(link = "identity"), data = train)
  pred.lm <- predict(lm.fit, newdata = test)
  logp(paste(label, "- linear regression"), summary(lm.fit))

  ## (2) CART：用 10-fold CV 的 xerror 最小者剪枝
  set.seed(seed)
  tree.full <- rpart(fml, data = train, method = "anova")
  cp.tab  <- tree.full$cptable
  cp.best <- cp.tab[which.min(cp.tab[, "xerror"]), "CP"]
  tree.fit  <- prune(tree.full, cp = cp.best)
  pred.tree <- predict(tree.fit, newdata = test)
  logp(paste(label, "- CART cptable"), cp.tab)
  save_base(paste0(prefix, "1_cart_tree.png"),
            rpart.plot(tree.fit, type = 2, extra = 101, roundint = FALSE,
                       box.palette = "Oranges",
                       main = sprintf("%s: Regression Tree (pruned, cp = %.4f)",
                                      label, cp.best)),
            w = 9, h = 6)

  ## (3) Random Forest：用 OOB MSE 選 mtry
  p <- length(xvars)
  mtry.grid <- seq_len(p)
  oob.mtry <- sapply(mtry.grid, function(m) {
    set.seed(seed)
    tail(randomForest(fml, data = train, ntree = 500, mtry = m)$mse, 1)
  })
  mtry.best <- mtry.grid[which.min(oob.mtry)]
  set.seed(seed)
  rf.fit  <- randomForest(fml, data = train, ntree = 500,
                          mtry = mtry.best, importance = TRUE)
  pred.rf <- predict(rf.fit, newdata = test)
  logp(paste(label, "- random forest"), rf.fit)

  ## (4) XGBoost：從訓練資料切 20% 驗證集選 max_depth
  x.all <- model.matrix(fml, data = rbind(train, test))[, -1, drop = FALSE]
  idx.tr <- seq_len(nrow(train))
  x.tr <- x.all[idx.tr, , drop = FALSE]
  x.te <- x.all[-idx.tr, , drop = FALSE]
  dtrain <- xgb.DMatrix(data = x.tr, label = y.tr)
  dtest  <- xgb.DMatrix(data = x.te, label = y.te)

  params <- list(objective = "reg:squarederror", eval_metric = "rmse",
                 max_depth = 3, eta = 0.1, subsample = 0.8,
                 colsample_bytree = 0.8, tree_method = "hist")
  set.seed(seed)
  val.idx <- sample(idx.tr, size = floor(0.2 * length(idx.tr)))
  d.inner <- xgb.DMatrix(data = x.tr[-val.idx, , drop = FALSE], label = y.tr[-val.idx])
  d.valid <- xgb.DMatrix(data = x.tr[val.idx, , drop = FALSE],  label = y.tr[val.idx])
  depth.grid <- 1:8
  val.rmse <- sapply(depth.grid, function(d) {
    pr <- params
    pr$max_depth <- d
    set.seed(seed)
    f <- xgb.train(params = pr, data = d.inner, nrounds = 200, verbose = 0)
    sqrt(mean((predict(f, newdata = d.valid) - y.tr[val.idx])^2))
  })
  depth.best <- depth.grid[which.min(val.rmse)]
  params$max_depth <- depth.best
  set.seed(seed)
  xgb.fit  <- xgb.train(params = params, data = dtrain, nrounds = 200, verbose = 0)
  pred.xgb <- predict(xgb.fit, newdata = dtest)

  ## 調參結果
  tune <- list(cp = cp.best, mtry = mtry.best, max_depth = depth.best)
  logp(paste(label, "- tuning: mtry vs OOB MSE"), data.frame(mtry = mtry.grid, OOB_MSE = oob.mtry))
  logp(paste(label, "- tuning: max_depth vs validation RMSE"), data.frame(max_depth = depth.grid, Valid_RMSE = val.rmse))
  logp(paste(label, "- chosen hyper-parameters"), unlist(tune))

  df.oob <- data.frame(ntree = seq_along(rf.fit$mse), mse = rf.fit$mse)
  p.oob <- ggplot(df.oob, aes(ntree, mse)) +
    geom_line(colour = unname(col_model["Random Forest"]), linewidth = 0.9) +
    labs(title = "RF: OOB MSE vs. trees", subtitle = paste("mtry =", mtry.best),
         x = "Number of trees", y = "OOB MSE") + theme_hw
  df.mtry <- data.frame(mtry = mtry.grid, oob = oob.mtry)
  p.mtry <- ggplot(df.mtry, aes(mtry, oob)) +
    geom_line(colour = "grey60") +
    geom_point(colour = unname(col_model["Random Forest"]), size = 2.5) +
    geom_point(data = df.mtry[df.mtry$mtry == mtry.best, ], shape = 21,
               size = 5, colour = "red", stroke = 1.2) +
    scale_x_continuous(breaks = mtry.grid) +
    labs(title = "RF: effect of mtry", subtitle = paste("best mtry =", mtry.best),
         x = "mtry", y = "OOB MSE") + theme_hw
  df.depth <- data.frame(depth = depth.grid, rmse = val.rmse)
  p.depth <- ggplot(df.depth, aes(depth, rmse)) +
    geom_line(colour = "grey60") +
    geom_point(colour = unname(col_model["XGBoost"]), size = 2.5) +
    geom_point(data = df.depth[df.depth$depth == depth.best, ], shape = 21,
               size = 5, colour = "red", stroke = 1.2) +
    scale_x_continuous(breaks = depth.grid) +
    labs(title = "XGBoost: effect of max_depth",
         subtitle = paste("best max_depth =", depth.best),
         x = "max_depth", y = "Validation RMSE") + theme_hw
  save_gg(arrangeGrob(p.oob, p.mtry, p.depth, ncol = 3),
          paste0(prefix, "2_tuning.png"), w = 15, h = 4.5)

  ## 變數重要性
  imp.cart <- tree.fit$variable.importance
  imp.rf   <- importance(rf.fit, type = 1)[, 1]
  imp.xgb  <- as.data.frame(xgb.importance(model = xgb.fit))
  logp(paste(label, "- CART importance"), imp.cart)
  logp(paste(label, "- RF importance (%IncMSE)"), sort(imp.rf, decreasing = TRUE))
  logp(paste(label, "- XGBoost importance"), imp.xgb)
  save_gg(arrangeGrob(
    imp_bar(imp.cart, "CART", "Importance", col_model["CART"]),
    imp_bar(imp.rf, "Random Forest", "%IncMSE", col_model["Random Forest"]),
    imp_bar(setNames(imp.xgb$Gain, imp.xgb$Feature), "XGBoost", "Gain", col_model["XGBoost"]),
    ncol = 3), paste0(prefix, "3_importance.png"), w = 14, h = 4.5)

  ## 測試資料表現
  model.levels <- names(col_model)
  preds <- list(pred.lm, pred.tree, pred.rf, pred.xgb)
  met <- do.call(rbind, lapply(seq_along(preds), function(i) {
    data.frame(Dataset = label, Model = model.levels[i], t(metrics(as.numeric(preds[[i]]), y.te)))
  }))
  logp(paste(label, "- test performance"), met)

  df.pred <- data.frame(
    Model = factor(rep(model.levels, each = length(y.te)), levels = model.levels),
    Predicted = unlist(lapply(preds, as.numeric)),
    Observed = rep(y.te, times = length(preds)))
  lab <- data.frame(Model = factor(model.levels, levels = model.levels),
                    x = lims[1], y = lims[2],
                    txt = sprintf("RMSE = %.3f\nR\u00b2 = %.3f", met$RMSE, met$R2))
  p.pred <- ggplot(df.pred, aes(Predicted, Observed, colour = Model)) +
    geom_abline(intercept = 0, slope = 1, linetype = 2, colour = "red") +
    geom_point(alpha = 0.75, size = 2) +
    geom_text(data = lab, aes(x = x, y = y, label = txt), inherit.aes = FALSE,
              hjust = 0, vjust = 1, size = 3.6) +
    facet_wrap(~ Model, ncol = 2) +
    scale_colour_manual(values = col_model, guide = "none") +
    coord_equal(xlim = lims, ylim = lims) +
    labs(title = paste0(label, ": Predicted vs. Observed (test set)")) +
    theme_hw
  save_gg(p.pred, paste0(prefix, "4_pred_obs.png"), w = 8, h = 8)

  met$Model <- factor(met$Model, levels = model.levels)
  p.met <- ggplot(met, aes(Model, RMSE, fill = Model)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = sprintf("%.3f", RMSE)), vjust = -0.4, size = 4.2) +
    scale_fill_manual(values = col_model, guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(title = paste0(label, ": Test RMSE by model"), x = NULL, y = "Test RMSE") +
    theme_hw
  save_gg(p.met, paste0(prefix, "5_rmse.png"), w = 7, h = 4.5)

  write.csv(met, fig(paste0(prefix, "_metrics.csv")), row.names = FALSE)
  invisible(list(metrics = met, tune = tune))
}


#############################################################################
## Part B. Boston Housing (Regression)
#############################################################################
data("BostonHousing", package = "mlbench")
set.seed(123)
split <- sample.split(BostonHousing$medv, SplitRatio = 0.8)
train_data <- subset(BostonHousing, split == TRUE)
test_data  <- subset(BostonHousing, split == FALSE)
logp("Boston train / test dim", rbind(train = dim(train_data), test = dim(test_data)))

res.boston <- run_regression(
  train = train_data, test = test_data, yname = "medv",
  xvars = setdiff(names(BostonHousing), "medv"),
  prefix = "B", label = "Boston Housing", lims = c(0, 50))


#############################################################################
## Part C. Baseball (Regression, y = Logsalary)
#############################################################################
Baseball <- read.table("ball.txt", header = TRUE, sep = "", quote = "\"", dec = ".")
cat.vars <- c("League86", "Div86", "Team86", "Pos86", "League87", "Team87")
Baseball[cat.vars] <- lapply(Baseball[cat.vars], factor)
logp("Baseball dim", dim(Baseball))
logp("Baseball summary of Salary / Logsalary", summary(Baseball[, c("Salary", "Logsalary")]))

## C-1 薪資分布：取 log 的理由
p.h1 <- ggplot(Baseball, aes(Salary)) +
  geom_histogram(bins = 30, fill = "grey60", colour = "white") +
  labs(title = "Salary (thousand USD)", x = NULL, y = "Count") + theme_hw
p.h2 <- ggplot(Baseball, aes(Logsalary)) +
  geom_histogram(bins = 30, fill = unname(col_model["XGBoost"]), colour = "white") +
  labs(title = "Logsalary = log(Salary)", x = NULL, y = "Count") + theme_hw
p.h3 <- ggplot(Baseball, aes(Yrs, Logsalary)) +
  geom_point(colour = unname(col_model["Random Forest"]), alpha = 0.7) +
  geom_smooth(method = "loess", formula = y ~ x, se = FALSE, colour = "red") +
  labs(title = "Years in MLB vs. Logsalary", x = "Yrs", y = "Logsalary") + theme_hw
save_gg(arrangeGrob(p.h1, p.h2, p.h3, ncol = 3), "C0_baseball_eda.png", w = 14, h = 4.2)

## C-2 訓練 / 測試資料
## 注意：Salary 是 Logsalary 的原始值，不能放進解釋變數 (會資訊外洩)
set.seed(123)
split.ball <- sample.split(Baseball$Logsalary, SplitRatio = 0.8)
train.ball <- subset(Baseball, split.ball == TRUE)
test.ball  <- subset(Baseball, split.ball == FALSE)
logp("Baseball train / test dim", rbind(train = dim(train.ball), test = dim(test.ball)))

## 解釋變數與原本 CART 作業 (fit1) 相同
x.ball <- c("Bat86", "Hit86", "Hr86", "Run86", "Rb86", "Wlk86", "Yrs",
            "Batcr", "Hitcr", "Hrcr", "Runcr", "Rbcr", "Wlkcr",
            "League86", "Div86", "Puto86", "Asst86", "Err86", "League87")

res.ball <- run_regression(
  train = train.ball, test = test.ball, yname = "Logsalary",
  xvars = x.ball, prefix = "C", label = "Baseball", lims = c(4, 8))


#############################################################################
## 總結
#############################################################################
all.metrics <- rbind(res.boston$metrics, res.ball$metrics)
logp("All regression results", all.metrics)
write.csv(all.metrics, fig("all_metrics.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = fig("sessionInfo.txt"))
cat("\n完成！所有圖檔與結果在：", normalizePath(out_dir), "\n")
