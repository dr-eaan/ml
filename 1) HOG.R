setwd('C:\\Users\\proto\\Desktop\\***** project')
#dir = file.choose()
#unzip(dir)
#setwd('C:/Users/proto/Desktop/***** project')
Sys.setenv(LANG = "en")
#install.packages('OpenImageR')
#install.packages('parallelSVM')
library(OpenImageR)
library(e1071)
library(caret)
library(Matrix)
library(png)
library(jpeg)
train = read.csv('Train.csv', stringsAsFactors = FALSE) #FULL
set.seed(42) 
train = train[sample(1:nrow(train), 4000, replace = FALSE),] #SAMPLE

baltrain = data.frame() #BALANCED
h = min(table(train$ClassId))
for (i in 0:42){
  t = table(train$ClassId)[i + 1]
  z = ceiling(h / (t/30))
  for (k in seq(from = (30-z+1), to = (t-z+1), by = 30)){
    baltrain = rbind(baltrain, train[which(train$ClassId == i)[(k:(k+z-1))],])
  }
}
train = baltrain
table(train$ClassId)

#HOG
#Train

trpic = vector(mode = 'list', length = nrow(train))
trpicol = vector(mode = 'list', length = nrow(train))

errors = vector(mode = 'list', length=nrow(train))

for (i in 1:nrow(train)){
  tryCatch({
    path = file.path(getwd(), train$Path[i])
    pic = readImage(path)
    n_w = train$Roi.X1[i]:train$Roi.X2[i]
    n_h = train$Roi.Y1[i]:train$Roi.Y2[i]
    pic = cropImage(pic, new_width = n_w, new_height = n_h, type = 'user_defined')
    pic = resizeImage(pic, 50, 50, method = 'bilinear') 
    pic_col = pic
    pic = rgb_2gray(pic)
    trpic[[i]] = pic
    trpicol[[i]] = pic_col
  }, error = function(e){
    errors[i] = i
  })
}

e = vector()
k = 1
for (i in 1:length(trpic)){
  if (is.null(trpic[[i]]) == TRUE){
    e[k] = i
    k = k + 1
  }
}
trpic = trpic[-(which(sapply(trpic, is.null), arr.ind = TRUE))]
trpicol = trpicol[-(which(sapply(trpicol, is.null), arr.ind = TRUE))]
saveRDS(trpicol, file = 'trcol.rds')

hogtrtime = c()
tmptrhog = vector(mode = 'list', length = length(trpic))
for (i in 1:length(trpic)){
  t1 = Sys.time()
  tmptrhog[[i]] = HOG(trpic[[i]], orientations = 16)
  t2 = Sys.time()
  hogtrtime[i] = t2 - t1
}
sum(hogtrtime)

trhog = matrix(0, nrow = length(tmptrhog), ncol = length(tmptrhog[[1]]))
tridhog = factor(train$ClassId, levels = 0:42)
tridhog = tridhog[-e]
#table(tridhog)
saveRDS(tridhog, file = 'tridhog.rds')

for (r in 1:length(tmptrhog)){
  for (c in 1:length(tmptrhog[[1]])){
    trhog[r,c] = tmptrhog[[r]][c]
  }
}

t1 = Sys.time() #RUN THESE 3 LINES TOGETHER
trmod = svm(trhog, tridhog)
t2 = Sys.time()
svmtime = t2 - t1
predSVM = predict(trmod)
confusionMatrix(predSVM, tridhog)

rm(errors, pic, tmptrhog, train, trhog) 
gc()

#Test
test = read.csv('Test.csv', stringsAsFactors = FALSE)
#test = test[sample(1:nrow(test), 1000, replace = FALSE),]

baltest = data.frame()
h = min(table(test$ClassId))
for (i in 0:42){
  t = table(test$ClassId)[i + 1]
  baltest = rbind(baltest, test[which(test$ClassId == i)[(t-h+1):t],])
}
test = baltest

tspic = vector(mode = 'list', length = nrow(test))
tspicol = vector(mode = 'list', length = nrow(test))

errors = vector(mode = 'list', length=nrow(test))

for (i in 1:nrow(test)){
  tryCatch({
    path = file.path(getwd(), test$Path[i])
    pic = readImage(path)
    n_w = test$Roi.X1[i]:test$Roi.X2[i]
    n_h = test$Roi.Y1[i]:test$Roi.Y2[i]
    pic = cropImage(pic, new_width = n_w, new_height = n_h, type = 'user_defined')
    pic = resizeImage(pic, 50, 50, method = 'bilinear')
    pic_col = pic
    pic = rgb_2gray(pic)
    #train$hog[[i]] = HOG(pic, orientations = 16)
    tspic[[i]] = pic
    tspicol[[i]] = pic_col
  }, error = function(e){
    errors[i] = i
  })
}

e = vector()
k = 1
for (i in 1:length(tspic)){
  if (is.null(tspic[[i]]) == TRUE){
    e[k] = i
    k = k + 1
  }
}
tspic = tspic[-(which(sapply(tspic, is.null), arr.ind = TRUE))]
tspicol = tspicol[-(which(sapply(tspicol, is.null), arr.ind = TRUE))]
saveRDS(tspicol, file = 'tscol.rds')

hogtstime = c()
tmptshog = vector(mode = 'list', length = length(tspic))
for (i in 1:length(tspic)){
  t1 = Sys.time()
  tmptshog[[i]] = HOG(tspic[[i]], orientations = 16)
  t2 = Sys.time()
  hogtstime[i] = t2 - t1
}
sum(hogtstime) 

tshog = matrix(0, nrow = length(tmptshog), ncol = length(tmptshog[[1]]))
tsidhog = factor(test$ClassId, levels = 0:42)
tsidhog = tsidhog[-e]
table(tsidhog)
saveRDS(tsidhog, file = 'tsidhog.rds')

for (r in 1:length(tmptshog)){
  for (c in 1:length(tmptshog[[1]])){
    tshog[r,c] = tmptshog[[r]][c]
  }
}

predSVM2 = predict(trmod, tshog)
confusionMatrix(predSVM2, tsidhog) 

rm(errors, pic, tmptshog, test, tshog)
gc()
