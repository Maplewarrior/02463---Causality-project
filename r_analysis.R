# load data
data<-read.csv(file = "data/data_98_observational.csv",header=T)

par(mfrow=c(2,3))
hist(data$A)
hist(data$B)
hist(data$C)
hist(data$D)
hist(data$E)
hist(data$F)

# load data
data<-read.csv(file = "test_data/data_30_A0.csv",header=T)

par(mfrow=c(2,3))
hist(data$A)
hist(data$B)
hist(data$C)
hist(data$D)

