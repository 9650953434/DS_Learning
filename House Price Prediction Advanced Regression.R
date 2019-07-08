library(ggplot2)
library(corrplot)
library(plyr)
library(randomForest)
library(gridExtra)
library(e1071)
library(caret)
library(xgboost)

setwd("C:/Users/ram/Desktop/Ram/Kaggle/house-prices-advanced-regression-techniques")
train<-read.csv("./train.csv", stringsAsFactors = FALSE)
test<-read.csv("./test.csv", stringsAsFactors = FALSE)
test_id<-test$Id
train_id<-train$Id
train$Id<-NULL
test$Id<-NULL
test$SalePrice<-NA
data_all<-rbind(train,test)
summary(data_all)

ggplot(data=data_all[!is.na(data_all$SalePrice),], aes(x=SalePrice))+
  geom_histogram(fill="blue", col="red", binwidth = 20000)+
  scale_x_continuous(breaks=seq(0,800000, by=100000))+
  ylab(label = "Count")+ xlab(label = "House Sales Price")

summary(data_all$SalePrice)
numeric_col<- names(which(sapply(data_all,is.numeric)))
cor_val<-cor(data_all[,numeric_col],use="pairwise.complete.obs")
View(cor_val)
cor_val_sorted<-as.matrix(sort(cor_val[,'SalePrice'],decreasing = TRUE))
View(cor_val_sorted)
corhigh<-names(which(apply(cor_val_sorted, 1, function(x) abs(x)>0.5)))
View(corhigh)

cor_val<-cor_val[corhigh, corhigh]
corrplot.mixed(cor_val,tl.pos = "lt", tl.col="black")

ggplot(data=data_all[!is.na(data_all$SalePrice),], aes(x=factor(OverallQual), y=SalePrice))+
  geom_boxplot(col="blue")+
  geom_point(col="red")+
  geom_smooth(method="auto", aes(group=1))+
  xlab("Overall Quality")+
  ylab("Home Sale Price")

ggplot(data=data_all[!is.na(data_all$SalePrice),], aes(x=GrLivArea,y=SalePrice))+
  geom_point(col="blue")+
  geom_smooth(method = "lm", col="red")+
  xlab("Living Area")+
  ylab("Home Sale Price")

ggplot(data=data_all[!is.na(data_all$SalePrice),], aes(x=factor(GarageCars), y=SalePrice))+
  #geom_point(col="blue")+
  geom_boxplot(col="blue")+
  xlab("Garage Area")+
  ylab("Home Sale Price")


col_na<-which(colSums(is.na(data_all))>0)
sort(colSums(sapply(data_all[col_na], is.na)), decreasing = TRUE) #--------------

###### HANDELING NA VALUES FOR POOLQC BY REPLACING NA VALUE WITH NONE #########################
data_all$PoolQC[is.na(data_all$PoolQC)]<-'None'
summary(as.factor(data_all$PoolQC))
Quality<-c('None'=0,'Po'=1,'Fa'=2,'TA'=3,'Gd'=4,'Ex'=5)
data_all$PoolQC<-as.integer(revalue(data_all$PoolQC,Quality))
summary(as.factor(data_all$PoolQC))

############ VALIDATING POOL AREA AND POOL QC CONSISTENCY AND IMPUTING NON CONSISTENT PAIR ######
data_all[data_all$PoolQC==0 & data_all$PoolArea>0,c('PoolQC', 'PoolArea', 'OverallQual') ]

summary(as.factor(data_all$OverallQual))
data_all$PoolQC[2421]<-2
data_all$PoolQC[2504]<-3
data_all$PoolQC[2600]<-2

############### HANDELING NA VALUE FOR MiscFeature ##########
summary(as.factor(data_all$MiscFeature))

data_all$MiscFeature[is.na(data_all$MiscFeature)]<-'None'

ggplot(data=data_all[!is.na(data_all$SalePrice),], aes(x=factor(MiscFeature),y=SalePrice))+
  geom_bar(stat = 'summary', fun.y="median", fill="blue")

############### HANDELING NA VALUE FOR Alley ##########
summary(as.factor(data_all$Alley))
data_all$Alley[is.na(data_all$Alley)]<-'None'


############### HANDELING NA VALUE FOR Fence ##########
summary(as.factor(data_all$Fence))
data_all$Fence[is.na(data_all$Fence)]<-'None'


############### HANDELING NA VALUE FOR FireplaceQu ##########
summary(as.factor(data_all$FireplaceQu))
data_all$FireplaceQu[is.na(data_all$FireplaceQu)]<-'None'
data_all$FireplaceQu<-as.integer(revalue(data_all$FireplaceQu,Quality))
data_all$FireplaceQu[is.na(data_all$FireplaceQu)]<-3

############### HANDELING NA VALUE FOR LotFrontage ##########
summary((data_all$LotFrontage))
for (i in 1:nrow(data_all)) 
{
  if(is.na(data_all$LotFrontage[i]))
  {
    data_all$LotFrontage[i]<-median(data_all$LotFrontage[data_all$Neighborhood==data_all$Neighborhood[i]],na.rm = TRUE)
  }
}

############### HANDELING NA VALUE FOR GarageYrBlt ##########
summary(as.factor(data_all$GarageYrBlt))
data_all$GarageYrBlt[is.na(data_all$GarageYrBlt)] <- data_all$YearBuilt[is.na(data_all$GarageYrBlt)]

length(which(is.na(data_all$GarageFinish) & is.na(data_all$GarageQual) & is.na(data_all$GarageCond)&is.na(data_all$GarageType)))

data_all[is.na(data_all$GarageFinish) & !is.na(data_all$GarageType),c('GarageCars','GarageArea', 'GarageFinish','GarageQual','GarageCond','GarageType')]
names(sort(table(data_all$GarageCond),decreasing = TRUE))[1]
data_all$GarageCond[2127]<-names(sort(table(data_all$GarageCond),decreasing = TRUE))[1]
data_all$GarageFinish[2127]<-names(sort(table(data_all$GarageFinish),decreasing = TRUE))[1]
data_all$GarageQual[2127]<-names(sort(table(data_all$GarageQual),decreasing = TRUE))[1]

data_all[2127,c('GarageCars','GarageArea', 'GarageFinish','GarageQual','GarageCond','GarageType')]
data_all$GarageCars[2577] <- 0
data_all$GarageArea[2577] <- 0
data_all$GarageType[2577]<-NA
data_all[2577,c('GarageCars','GarageArea', 'GarageFinish','GarageQual','GarageCond','GarageType')]
summary(factor(data_all$GarageType))
data_all$GarageType[is.na(data_all$GarageType)]<-'No Garage'

summary(factor(data_all$GarageFinish))
data_all$GarageFinish[is.na(data_all$GarageFinish)]<-'None'
Finish <- c('None'=0, 'Unf'=1, 'RFn'=2, 'Fin'=3)
data_all$GarageFinish<-as.integer(revalue(data_all$GarageFinish,Finish))

summary(factor(data_all$GarageQual))
data_all$GarageQual[is.na(data_all$GarageQual)]<-'None'
#data_all$GarageQual[data_all$GarageQual=='TA']<-'Ta'
Quality
data_all$GarageQual<-as.integer(revalue(data_all$GarageQual, Quality))

summary(factor(data_all$GarageCond))
data_all$GarageCond[is.na(data_all$GarageCond)]<-'None'
data_all$GarageCond<-as.integer(revalue(data_all$GarageCond, Quality))

length(which(is.na(data_all$BsmtCond) & is.na(data_all$BsmtExposure) & 
               is.na(data_all$BsmtQual) & 
               is.na(data_all$BsmtFinType2) & 
               is.na(data_all$BsmtFinType1)))


data_all[!is.na(data_all$BsmtFinType1) & (is.na(data_all$BsmtCond) | is.na(data_all$BsmtQual)|is.na(data_all$BsmtExposure)|is.na(data_all$BsmtFinType2)), 
                                          c('BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2','BsmtFinType1')]

names(sort(table(data_all$BsmtFinType2), decreasing = TRUE)[1])
data_all$BsmtFinType2[333]<-names(sort(table(data_all$BsmtFinType2), decreasing = TRUE)[1])

data_all$BsmtExposure[c(949,1488,2349)]<-names(sort(table(data_all$BsmtExposure), decreasing = TRUE)[1])
data_all$BsmtQual[c(2218,2219)]<-names(sort(table(data_all$BsmtQual), decreasing = TRUE)[1])
data_all$BsmtCond[c(2041, 2186, 2525)]<-names(sort(table(data_all$BsmtCond), decreasing = TRUE)[1])

summary(factor(data_all$BsmtQual))
data_all$BsmtQual[is.na(data_all$BsmtQual)]<-'None'
data_all$BsmtQual<-as.integer(revalue(data_all$BsmtQual,Quality))

summary(factor(data_all$BsmtCond))
data_all$BsmtCond[is.na(data_all$BsmtCond)]<-'None'
data_all$BsmtCond<-as.integer(revalue(data_all$BsmtCond,Quality))

summary(factor(data_all$BsmtExposure))
data_all$BsmtExposure[is.na(data_all$BsmtExposure)]<-'None'
Exposure<-c('None'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)
data_all$BsmtExposure<-as.integer(revalue(data_all$BsmtExposure,Exposure))

summary(factor(data_all$BsmtFinType1))
data_all$BsmtFinType1[is.na(data_all$BsmtFinType1)]<-'None'
FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
data_all$BsmtFinType1<-as.integer(revalue(data_all$BsmtFinType1,FinType))

summary(factor(data_all$BsmtFinType2))
data_all$BsmtFinType2[is.na(data_all$BsmtFinType2)]<-'None'
#FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
data_all$BsmtFinType2<-as.integer(revalue(data_all$BsmtFinType2,FinType))

####################### MasVnrType, MasVnrArea #################################
length(data_all[is.na(data_all$MasVnrType)& is.na(data_all$MasVnrArea),c('MasVnrType', 'MasVnrArea')])
data_all[is.na(data_all$MasVnrType)& !is.na(data_all$MasVnrArea),c('MasVnrType', 'MasVnrArea')]

summary(factor(data_all$MasVnrType))
data_all$MasVnrType[2611]<-names(sort(table(data_all$MasVnrType), decreasing = TRUE)[2])

summary(factor(data_all$MasVnrType))
data_all$MasVnrType[is.na(data_all$MasVnrType)]<-'None'
Masonry <- c('None'=0, 'BrkCmn'=0, 'BrkFace'=1, 'Stone'=2)
data_all$MasVnrType<-as.integer(revalue(data_all$MasVnrType,Masonry))

summary(data_all$MasVnrArea)
data_all$MasVnrArea[is.na(data_all$MasVnrArea)]<-0


data_all[is.na(data_all$MSZoning), 'MSZoning']
summary(factor(data_all$MSZoning))
sort(table(data_all$MSZoning), decreasing = TRUE)
data_all$MSZoning[is.na(data_all$MSZoning)]<-names(sort(table(data_all$MSZoning), decreasing = TRUE)[1])


####################### KitchenQual #################################
summary(factor(data_all$KitchenQual))
data_all$KitchenQual[is.na(data_all$KitchenQual)]<-names(sort(table(data_all$KitchenQual), decreasing = TRUE)[1])

####################### Utilities #################################
data_all$Utilities <- NULL

####################### BsmtFullBath,BsmtHalfBath, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF  #################################
summary(factor(data_all$BsmtFullBath))
length(data_all$BsmtFullBath[is.na(data_all$BsmtHalfBath)])
data_all$BsmtFullBath[is.na(data_all$BsmtFullBath)]<-names(sort(table(data_all$BsmtFullBath), decreasing = TRUE)[1])
summary(factor(data_all$BsmtHalfBath))
data_all$BsmtHalfBath[is.na(data_all$BsmtHalfBath)]<-names(sort(table(data_all$BsmtHalfBath), decreasing = TRUE)[1])

summary(factor(data_all$BsmtFinSF1))
data_all$BsmtFinSF1[is.na(data_all$BsmtFinSF1)]<-0
data_all$BsmtFinSF2[is.na(data_all$BsmtFinSF2)]<-0

summary(factor(data_all$BsmtUnfSF))
data_all$BsmtUnfSF[is.na(data_all$BsmtUnfSF)]<-0
data_all$BsmtUnfSF[is.na(data_all$BsmtUnfSF)]<-0
data_all$TotalBsmtSF[is.na(data_all$TotalBsmtSF)]<-0
summary(factor(data_all$TotalBsmtSF))



####################### Functional  #################################
summary(factor(data_all$Functional))
data_all$Functional[is.na(data_all$Functional)] <- names(sort(-table(data_all$Functional)))[1]
data_all$Functional <- as.integer(revalue(data_all$Functional, c('Sal'=0, 'Sev'=1, 'Maj2'=2, 'Maj1'=3, 'Mod'=4, 'Min2'=5, 'Min1'=6, 'Typ'=7)))

####################### Exterior1st, Exterior2nd  #################################
summary(factor(data_all$Exterior1st))
summary(factor(data_all$Exterior2nd))
data_all$Exterior1st[is.na(data_all$Exterior1st)] <- names(sort(-table(data_all$Exterior1st)))[1]
data_all$Exterior2nd[is.na(data_all$Exterior2nd)] <- names(sort(-table(data_all$Exterior2nd)))[1]

data_all$ExterQual<-as.integer(revalue(data_all$ExterQual, Quality))
data_all$ExterCond<-as.integer(revalue(data_all$ExterCond, Quality))

####################### Electrical  #################################
summary(factor(data_all$Electrical))
data_all$Electrical[is.na(data_all$Electrical)] <- names(sort(-table(data_all$Electrical)))[1]

####################### SaleType  #################################
summary(factor(data_all$SaleType))
data_all$SaleType[is.na(data_all$SaleType)] <- names(sort(-table(data_all$SaleType)))[1]


str(data_all)
names(data_all[,sapply(data_all, is.character)])

summary(factor(data_all$HeatingQC))
data_all$HeatingQC<-as.integer(revalue(data_all$HeatingQC,Quality))

summary(factor(data_all$KitchenQual))
data_all$KitchenQual<-as.integer(revalue(data_all$KitchenQual,Quality))

summary(factor(data_all$CentralAir))
data_all$CentralAir<-as.integer(revalue(data_all$CentralAir,c('N'=0,'Y'=1)))
data_all$Street<-as.integer(revalue(data_all$Street, c('Grvl'=0, 'Pave'=1)))
data_all$PavedDrive<-as.integer(revalue(data_all$PavedDrive, c('N'=0, 'P'=1, 'Y'=2)))
summary(factor(data_all$BsmtFullBath))
data_all[sapply(data_all, is.character)] <- lapply(data_all[sapply(data_all, is.character)], as.factor)
data_all$BsmtHalfBath<-as.integer(data_all$BsmtHalfBath)
data_all$BsmtFullBath<-as.integer(data_all$BsmtFullBath)
data_all$MoSold <- as.factor(data_all$MoSold)

ggplot(data = data_all, aes(x=YrSold, y=SalePrice))+geom_bar(stat = 'summary',fun.y="median", fill="blue")

summary(factor(data_all$MSSubClass))
data_all$MSSubClass<-as.factor(data_all$MSSubClass)

########################################## Correlation Again ################################################
numeric_col_again<-data_all[,names(data_all[,sapply(data_all, is.numeric)])]
corr_again<-cor(numeric_col_again,use = 'pairwise.complete.obs')
corr_sorted<-as.matrix(sort(corr_again[,'SalePrice'],decreasing = TRUE))
corhigh<-names(which(abs(corr_sorted[,1])>0.5))

corrplot.mixed(corr_again[corhigh,corhigh],tl.pos = "lt", tl.col="black")

set.seed(2018)
View(data_all)

rf<-randomForest(x=data_all[1:1460,-79], y=data_all$SalePrice[1:1460],ntree = 100, importance = TRUE)
rf_imp<-importance(rf)
rm_imp<-data.frame(Variables=row.names(rf_imp),MSE=rf_imp[,1])
rm_imp<-rm_imp[order(rm_imp$MSE, decreasing =TRUE),]
rm_imp
ggplot(data=rm_imp[1:20,],aes(x=reorder(Variables, MSE),y=MSE, fill=MSE))+geom_bar(stat = 'identity')+ coord_flip()



str(data_all[,row.names(rm_imp[1:20,])]) ######only 3 categorial variable having high importance#########


#####################Feature analysis of high importance variables

##################### GrLivArea ############################

a1<-ggplot(data=data_all[!is.na(data_all$SalePrice),],aes(x=GrLivArea,y=SalePrice))+geom_point(col='blue')+
  geom_text(aes(label = ifelse(data_all$GrLivArea[!is.na(data_all$SalePrice)]>4500, rownames(data_all),'')))
a2<-ggplot(data=data_all,aes(x=GrLivArea))+geom_density()
grid.arrange(a1, a2, nrow = 1)

##################### Neighborhood ############################
b1<-ggplot(data=data_all[1:1460,],aes(x=factor(Neighborhood),y=SalePrice))+
  geom_bar(stat = "summary",fun.y="median",fill="blue")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  geom_label(stat = "count", aes(label=..count..,y=..count..))
b2<-ggplot(data=data_all, aes(x=factor(Neighborhood)))  +
  geom_histogram(stat="count", fill="green")+
  geom_label(stat = "count", aes(label=..count..,y=..count..))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
grid.arrange(b1, b2, ncol = 1)

#################################Total Number of Bathrooms###########################################
data_all$totalbathrooms<-data_all$BsmtFullBath+(data_all$BsmtHalfBath*0.5)+data_all$FullBath+(data_all$HalfBath*0.5)
p1<-ggplot(data = data_all[!is.na(data_all$SalePrice),], aes(x=totalbathrooms,y=SalePrice))+geom_point(col='blue')
p2<-ggplot(data=data_all, aes(x=as.factor(totalbathrooms)))+geom_histogram(stat = 'count')
grid.arrange(p1,p2,ncol=1)
cor(data_all[!is.na(data_all$SalePrice),'SalePrice'],data_all[!is.na(data_all$SalePrice),'totalbathrooms'])

################################Age of House#########################################################
data_all$remodeled<-ifelse(data_all$YearBuilt==data_all$YearRemodAdd,0,1)
#data_all$homeage<-data_all$YrSold - data_all$YearRemodAdd
data_all$homeage<-data_all$YrSold - data_all$YearBuilt

data_all$isnew<-ifelse(data_all$YrSold==data_all$YearBuilt,1,0)

ggplot(data = data_all[!is.na(data_all$SalePrice),],aes(x=as.factor(remodeled),y=SalePrice))+
  geom_bar(stat = 'summary',fun.y="median",fill='blue')+
  geom_label(stat = 'count',aes(label=..count..,y=..count..))

ggplot(data = data_all[!is.na(data_all$SalePrice),],aes(x=as.factor(isnew),y=SalePrice))+
  geom_bar(stat = 'summary',fun.y="median",fill='blue')+
  geom_label(stat = 'count',aes(label=..count..,y=..count..))

ggplot(data = data_all[!is.na(data_all$SalePrice),],aes(x=homeage,y=SalePrice))+
  geom_point(col='blue')+geom_smooth(method = 'lm',col='black')

cor(data_all$SalePrice[!is.na(data_all$SalePrice)],data_all$homeage[!is.na(data_all$SalePrice)])
############################# Neighbourhood ##########################################################
ggplot(data = data_all[!is.na(data_all$SalePrice),], aes(x= reorder(Neighborhood,SalePrice,FUN = median),y=SalePrice))+
  geom_bar(stat = 'summary',fun.y='median',fill='blue')+
  geom_label(stat ="count", aes(label=..count..,y=..count..))

ggplot(data = data_all[!is.na(data_all$SalePrice),], aes(x= reorder(Neighborhood,SalePrice,FUN = mean),y=SalePrice))+
  geom_bar(stat = 'summary',fun.y='mean',fill='blue')+
  geom_label(stat ="count", aes(label=..count..,y=..count..))

ggplot(data = data_all[!is.na(data_all$SalePrice),], aes(x= reorder(Neighborhood,SalePrice,FUN = median),y=SalePrice))+
  geom_point(col='blue')

data_all$neighrich[data_all$Neighborhood %in% c('StoneBr', 'NridgHt', 'NoRidge')]<-2
data_all$neighrich[!data_all$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale', 'StoneBr', 'NridgHt', 'NoRidge')]<-1
data_all$neighrich[data_all$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale')]<-0
summary(factor(data_all$neighrich))

###############################################total area##############################################################
data_all$totalsqfeet<-data_all$GrLivArea + data_all$TotalBsmtSF

ggplot(data = data_all[!is.na(data_all$SalePrice),], aes(x=totalsqfeet,y=SalePrice))+
  geom_point(col='blue')+ geom_smooth(method='lm', col='black', se=FALSE)+
  geom_text(aes(label=ifelse(data_all$totalsqfeet[!is.na(data_all$SalePrice)]>7500,rownames(data_all),'')))

cor(data_all$SalePrice[!is.na(data_all$SalePrice)],data_all$totalsqfeet[!is.na(data_all$SalePrice)])
cor(data_all$SalePrice[-c(524,1299)],data_all$totalsqfeet[-c(524,1299)], use = 'pairwise.complete.obs')

##############################################Dropping highly correlated variables ###################################
dropvars<-c('YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'TotalBsmtSF', 'TotalRmsAbvGrd', 'BsmtFinSF1')
data_all<-data_all[,!names(data_all) %in% dropvars]

#############################################total porch sf #########################################################
data_all$totalporchsf <- data_all$OpenPorchSF + data_all$EnclosedPorch + data_all$X3SsnPorch + data_all$ScreenPorch
cor(data_all$SalePrice, data_all$totalporchsf, use= "pairwise.complete.obs")

#############################################Removing outliers########################################################
data_all<-data_all[-c(524,1299),]


numeric_col <- numeric_col[!(numeric_col %in% c('MSSubClass', 'MoSold', 'YrSold', 'SalePrice', 'OverallQual', 'OverallCond'))] 
numeric_col <- append(numeric_col, c('homeage', 'totalporchsf', 'totalbathrooms', 'totalsqfeet'))



DFnumeric <- data_all[, names(data_all) %in% numeric_col]

DFfactors <- data_all[, !(names(data_all) %in% numeric_col)]
DFfactors <- DFfactors[, names(DFfactors) != 'SalePrice']

cat('There are', length(DFnumeric), 'numeric variables, and', length(DFfactors), 'factor variables')

length(data_all)

for(i in 1:ncol(DFnumeric)){
  if(abs(skewness(DFnumeric[,i]))>=0.8){
    DFnumeric[,i]<-log(DFnumeric[,i]+1)
  }
}
PreNum<-scale(DFnumeric,center = TRUE)
View(PreNum)  
dim(PreNum)

####################################Encoding Categorical Variable##############################################

DFdummies <- as.data.frame(model.matrix(~.-1, DFfactors))         ###############Creating dummy variables
DFdummies
dim(DFdummies)  

###########################Removing dummy variables with all zero value in test################################
zerocoltest<-which(colSums(DFdummies[(nrow(data_all[!is.na(data_all$SalePrice),])+1):nrow(data_all),])==0)
DFdummies<-DFdummies[,-zerocoltest]

###########################Removing dummy variables with all zero value in train################################
zerocoltrain<-which(colSums(DFdummies[1:nrow(data_all[!is.na(data_all$SalePrice),]),])==0)
DFdummies<-DFdummies[,-zerocoltrain]

###########################Removing dummy variables with few ones value in train################################
fewonestrain<-which(colSums(DFdummies[1:nrow(data_all[!is.na(data_all$SalePrice),]),])<10)
View(fewonestrain)
DFdummies<-DFdummies[,-fewonestrain]

combinedDF <- cbind(PreNum, DFdummies)
qqline(data_all$SalePrice)

data_all$SalePrice<-log(data_all$SalePrice)
skewness(data_all$SalePrice[!is.na(data_all$SalePrice)])
qqline(data_all$SalePrice)
qqnorm(data_all$SalePrice)

############################combining saleprice with test and train data ####################
train1 <- combinedDF[!is.na(data_all$SalePrice),]
test1 <- combinedDF[is.na(data_all$SalePrice),]

set.seed(13234)
train1<-data.matrix(train1, rownames.force = NA)


my_control <-trainControl(method="cv", 
                          number=5, 
                          savePredictions = TRUE 
                          )

lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))


lasso_mod <- train(x=train1, y=data_all$SalePrice[!is.na(data_all$SalePrice)], method='glmnet', 
                   trControl= my_control, tuneGrid=lassoGrid)
lasso_mod
nrow(train1)
View(train1)
View(data.frame(data_all$SalePrice[!is.na(data_all$SalePrice)]))
View(data_all)

lasso_mod$bestTune
min(lasso_mod$results$RMSE)
plot(log(lasso_mod$results$lambda), lasso_mod$results$RMSE)

lassoVarImp <- varImp(lasso_mod)
lassoVarImp
lassoImportance <- lassoVarImp$importance
LassoPred <- predict(lasso_mod, test1)
predictions_lasso <- exp(LassoPred)

################################################XGBOOST##################################################

my_control <-trainControl(method="cv", number=5)

xgb_grid = expand.grid(
  nrounds = 500,
  eta = c(0.1, 0.05, 0.01),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree=1,
  min_child_weight=c(1, 2, 3, 4 ,5),
  subsample=1
)

xgb_caret <- train(x=train1, 
                   y=data_all$SalePrice[!is.na(data_all$SalePrice)], 
                   method='xgbTree', 
                   trControl= my_control, 
                   tuneGrid=xgb_grid)

xgb_caret

min(xgb_caret$results$RMSE)




param_list<-list(eta=0.01, 
                 gamma=0, 
                 colsample_bytree=1, 
                 min_child_weight=5, 
                 max_depth=5,
                 subsample=1, 
                 objective="reg:linear",
                 booster="gbtree")
xgbtrain<-xgb.DMatrix(data=as.matrix(train1), label=data_all$SalePrice[!is.na(data_all$SalePrice)])
xgbtest<-xgb.DMatrix(data=as.matrix(test1))

xgb_cv<-xgb.cv(data=xgbtrain, 
               params = param_list,
               nrounds = 1500,
               nfold = 5, 
               showsd = T, 
               stratified = T,
               print_every_n = 20, 
               early_stopping_rounds = 10, 
               maximize = F)

xgb_error<-data.frame(xgb_cv$evaluation_log)
plot(xgb_error$iter,xgb_error$train_rmse_mean, col='blue')
lines(xgb_error$iter,xgb_error$test_rmse_mean, col='red')

min(xgb_cv$evaluation_log$test_rmse_mean)

xgb_mod<-xgb.train(params = param_list,nrounds = 1348, data=xgbtrain)

xgb_predict<-predict(xgb_mod, xgbtest)
xgb_predict<-exp(xgb_predict)
head(xgb_predict)

###################################FEATURE IMPORTANCE#######################################################
xgb_imp<-xgb.importance(colnames(xgbtrain), model = xgb_mod)
print(xgb_imp)
xgb.ggplot.importance(xgb_imp[1:20])

###################################Final model##############################################################
final_model<-data.frame(test_id, SalePrice=(xgb_predict+2*predictions_lasso)/3)
head(final_model)
colnames(final_model)[1]<-"Id"
write.csv(final_model,file = "Submission.csv",row.names = F)

