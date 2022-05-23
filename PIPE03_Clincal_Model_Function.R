set.seed(830)
### summary os
library(rms)
library(survminer) 
library(survival)
library(survivalsvm)
library(survminer)
library(dplyr)
library(glmnet)
library(VIM)


############### Summary One Features #############


Select_Features <- function(Input_P_Data){
	min_p_value <- c(1:length(Input_P_Data[,1]))
	for(i in c(1:length(Input_P_Data[,1]))){
		min_p_value[i] <- min(Input_P_Data[i,c(3,4,5)])
	}
	Input_P_Data$new_pvalue <- min_p_value
	Out_list <- as.vector(Input_P_Data[which(Input_P_Data$new_pvalue < 1),1])
	return(Out_list)
}

Select_Cor_Features <- function(Input_P_Data, Data_table){
	print('+++++++++++++Part One+++++++++++++++++')
	Data_table_Cor  = Data_table[,c(46:345)]
	Cor_table <- cor(Data_table_Cor)
        min_p_value <- c(1:length(Input_P_Data[,1]))
	min_type <- c(1:length(Input_P_Data[,1]))
        for(i in c(1:length(Input_P_Data[,1]))){
                min_p_value[i] <- min(Input_P_Data[i,c(3,4,5)])
        }
	for(i in c(1:length(Input_P_Data[,1]))){
		One_cor_list = as.vector(Cor_table[,i])
		Cor_ID_num_List = which(One_cor_list>0.85)
		Min_ID_num = which.min(min_p_value[which(One_cor_list>0.85)])
		Min_ID = Cor_ID_num_List[Min_ID_num]
		if(Min_ID==i&min_p_value[i] < 0.2){
			min_type[i] <- 'Z'
		}else{
			min_type[i] <- 'O'
		}
	}
        Input_P_Data$new_pvalue <- min_p_value
	Input_P_Data$type <- min_type
        #Out_list <- as.vector(Input_P_Data[which(Input_P_Data$new_pvalue < 0.05),1])
        Out_list <- as.vector(Input_P_Data[which(Input_P_Data$new_pvalue < 1 & Input_P_Data$type=='Z'),1])
	print('++++++++++++++++++++Filter Dim+++++++++++++++++++++++++')
	print(length(Out_list))
        return(Out_list)
}


Select_Data <- function(Data_table, P_value_Table, type_ID){
	One_select_list = Select_Cor_Features(P_value_Table,Data_table)
	Data_Feature = Data_table[,One_select_list]
	for(i in c(1:length(Data_Feature[1,]))){
                min_num = min(Data_Feature[,i])
                max_num = max(Data_Feature[,i])
                min_max = max_num - min_num
                Data_Feature[,i] <- (Data_Feature[,i] - min_num)/min_max
        }
	colnames(Data_Feature) <- paste0(colnames(Data_Feature), '_', type_ID)
	Data_Feature$ID <- Data_table$new_name
	print(dim(Data_Feature))
	return(Data_Feature)
}



Select_Clincal_Data <- function(Data_table){
	Clincal_Part_Data = Data_table[,c(7,8,17)]
	Clincal_Part_Data[which(is.na(Clincal_Part_Data[,3])),3] <- 0
	Clincal_Part_Data$ID = Data_table$new_name
	return(Clincal_Part_Data)
}

Select_Time_Data <- function(Data_table, Time_type){
	Input_Data = Data_table
	if(Time_type == 'PFS'){
		T_Data <- data.frame(Input_Data$progress, Input_Data$PFS, Data_table$new_name)
		colnames(T_Data) = c('State', 'OS', 'ID')
	}
	if(Time_type == 'OS'){
		T_Data <- data.frame(Input_Data$die, Input_Data$OS, Data_table$new_name)
                colnames(T_Data) = c('State', 'OS', 'ID')
	}
	if(Time_type == 'DMFS'){
                T_Data <- data.frame(Input_Data$metastasis, Input_Data$DMFS, Data_table$new_name)
                colnames(T_Data) = c('State', 'OS', 'ID')
        }
	return(T_Data)
}	


## OS
Sus_Summary <- function(Data_table_C, Data_table_T1, Data_table_T2, P_value_Table_C, P_value_Table_T1, P_value_Table_T2, Time_type, output_cox_predict, output_cox_path){
	Clincal_Data = Select_Clincal_Data(Data_table_C)
	Merege_Table = Merge_List_Data(Data_table_C, Data_table_T1, Data_table_T2, P_value_Table_C, P_value_Table_T1, P_value_Table_T2, Clincal_Data)
	Time_Data = Select_Time_Data(Data_table_C, Time_type)
	if(TRUE){
	Summary_Data = merge(Time_Data, Merege_Table, x.by ='ID', y.by = 'ID')
	Group_Table <- read.csv('ID_Group.csv')
	Group_Table_use_list <- as.vector(Group_Table[which(Group_Table[,2]%in%c('center3', 'center2')),1])
	Summary_Data_part = Summary_Data[which(Summary_Data$ID%in%Group_Table_use_list),]
	rownames(Summary_Data) = as.character(Summary_Data$ID)

	Summary_Data = Summary_Data[,-which(colnames(Summary_Data)=='ID')]
	dt <- Summary_Data[,-which(colnames(Summary_Data)%in%c('OS', 'State'))]
	x.factors <- model.matrix(~ .,dt)[,-1]
	x <- as.matrix(data.frame(x.factors,dt[,3]))
	y <- data.matrix(Surv(Summary_Data$OS, Summary_Data$State))
	set.seed(30)
	fit <-glmnet(x,y,family = "cox",alpha = 1)
	fitcv <- cv.glmnet(x,y,family="cox", alpha=1,nfolds=10)
	out_coef <- coef(fitcv, s="lambda.min")
	out_coef_data <- data.frame(rownames(out_coef), as.vector(out_coef))
	Summary_Data = Summary_Data[,-c(which(out_coef_data[,2]==0)+2)]
	Summary_Data$ID <- rownames(Summary_Data)
	Summary_Data = merge(Summary_Data, Clincal_Data, x.by ='ID', y.by = 'ID')	
	Summary_Data = Summary_Data[,-which(colnames(Summary_Data)=='ID')]
	Summary_Data = Summary_Data[,-which(colnames(Summary_Data)%in%c('T', 'N', 'LDH2'))]

	sdf <- coxph(Surv(OS, State) ~ ., Summary_Data)
	m <- sdf
	##################################

	beta <- coef(m)
	se <- sqrt(diag(vcov(m)))
	HR <- exp(beta)
	HRse <- HR * se
	p = 1 - pchisq((beta/se)^2, 1)
	Find_ID <- names(p)[which(p <2)]
	Find_ID_list <- c(Find_ID, 'OS', 'State')
	Summary_Data_New <- Summary_Data[,which(colnames(Summary_Data)%in%Find_ID_list)]
	sdf <- coxph(Surv(OS, State) ~ ., Summary_Data_New)
		print('++++++++++++++++++++++++++++++++++++++')
        pred_cox = predict(object = m, newdata = Summary_Data_part)
        pred_cox <- as.vector(pred_cox)
        one_median_pvalue <- ifelse(pred_cox > quantile(pred_cox)[3], 'High', 'Low')
        one_high_pvalue <- ifelse(pred_cox > quantile(pred_cox)[4], 'High', 'Low')
        one_low_pvalue <- ifelse(pred_cox > quantile(pred_cox)[2], 'High', 'Low')
		Summary_Data_part$median <- one_median_pvalue
        Summary_Data_part$high <- one_high_pvalue
        Summary_Data_part$low <- one_low_pvalue
        Summary_Data_part$pred_cox_value <- pred_cox
		high_sdf<-survdiff(formula = Surv(OS, State) ~ high, data = Summary_Data_part)
        low_sdf<-survdiff(formula = Surv(OS, State) ~ low, data = Summary_Data_part)
        median_sdf<-survdiff(formula = Surv(OS, State) ~ median, data = Summary_Data_part)
        high.p.val <- 1 - pchisq(high_sdf$chisq, length(high_sdf$n) - 1)
        low.p.val <- 1 - pchisq(low_sdf$chisq, length(low_sdf$n) - 1)
        median.p.val <- 1 - pchisq(median_sdf$chisq, length(median_sdf$n) - 1)
        Summary_Data_part$name <- rownames(Summary_Data_part)
        write.csv(Summary_Data_part, output_cox_predict, row.names = F)
		sum.surv <- summary(sdf)
        c_index <-sum.surv$concordance
        print(c_index)
		tmp <- out_coef_data
        write.csv(tmp, output_cox_path, row.names = T)
		return(Summary_Data_part)
	}
}




Sus_Summary_plot <- function(Data_Input, Out_Pdf){
	Data_Input$OS <- Data_Input$OS*30
	print(summary(Data_Input$OS))
	coxm_1 <- cph(Surv(OS, State)~pred_cox_value, data=Data_Input,surv=T,x=T,y=T,time.inc = 1000)

	cal_1<-calibrate(coxm_1,u=1000,cmethod='KM', method = 'boot',m=200, B=500)
	pdf(Out_Pdf)
	par(mar=c(7,4,4,3),cex=1.0)
	plot(cal_1,lwd=2,lty=1, 
	     errbar.col=c(rgb(0,118,192,maxColorValue = 255)), 
	     xlab='Nomogram-Predicted Probability of 70 month OS/PFS/DMFS',
	     ylab='Actual 70 months OS/PFS/DMFS(proportion)',
	     col=c(rgb(192,98,83,maxColorValue = 255)),
	     xlim = c(0.8,1),ylim = c(0.8,1)) 
	dev.off()
}


Sus_plot <- function(Data_Input, Out_Pdf){
	fit <- survfit(Surv(OS, State)~high,data=Data_Input)
	pdf(Out_Pdf)
	p <- ggsurvplot(fit,data=Data_Input,
		   pval = TRUE, 
	           risk.table = TRUE,
        	   risk.table.col = "strata",
	           surv.median.line = "hv",
        	   ggtheme = theme_bw(),
	           legend.labs = c("low", "high"),
        	   palette = c("#FF6103", "#3D9140"))
	print(p)
	dev.off()
}

surv_summary <- function(Input_Data, One_select_list, select_type, output_path, output_cox_path, output_cox_predict){
	Data_Feature = Input_Data[,One_select_list]
	print(dim(Data_Feature))
	for(i in c(1:length(Data_Feature[1,]))){
		min_num = min(Data_Feature[,i])
		max_num = max(Data_Feature[,i])
		min_max = max_num - min_num
		Data_Feature[,i] <- (Data_Feature[,i] - min_num)/min_max
	}
	if(select_type == 'OS'){
		OS_Data = data.frame(Input_Data$die, Input_Data$OS)
	}
	if(select_type == 'DMFS'){
                #OS_Data = data.frame(Input_Data$die, Input_Data$OS)
		OS_Data = data.frame(Input_Data$metastasis, Input_Data$DMFS)
        }
	if(select_type == 'PFS'){
                #OS_Data = data.frame(Input_Data$die, Input_Data$OS)
		OS_Data = data.frame(Input_Data$progress, Input_Data$PFS)
        }
	colnames(OS_Data) = c('State', 'OS')
	sus_data = cbind(OS_Data, Data_Feature)
	sdf <- coxph(Surv(OS, State) ~ ., data = sus_data)
	print('+++++++++++++++++++++++++COX+++++++++++++++++++')
	m <- sdf
	pred_cox = predict(object = m, newdata = sus_data)
	pred_cox <- as.vector(pred_cox)
	one_median_pvalue <- ifelse(pred_cox > quantile(pred_cox)[3], 'High', 'Low')
        one_high_pvalue <- ifelse(pred_cox > quantile(pred_cox)[4], 'High', 'Low')
        one_low_pvalue <- ifelse(pred_cox > quantile(pred_cox)[2], 'High', 'Low')
	sus_data_n <- sus_data
	sus_data$median <- one_median_pvalue
	sus_data$high <- one_high_pvalue
	sus_data$low <- one_low_pvalue
	sus_data$pred_cox_value <- pred_cox
	high_sdf<-survdiff(formula = Surv(OS, State) ~ high, data = sus_data)
        low_sdf<-survdiff(formula = Surv(OS, State) ~ low, data = sus_data)
        median_sdf<-survdiff(formula = Surv(OS, State) ~ median, data = sus_data)
        high.p.val <- 1 - pchisq(high_sdf$chisq, length(high_sdf$n) - 1)
        print(high.p.val)
        low.p.val <- 1 - pchisq(low_sdf$chisq, length(low_sdf$n) - 1)
        print(low.p.val)
        median.p.val <- 1 - pchisq(median_sdf$chisq, length(median_sdf$n) - 1)
        print(median.p.val)
	sus_data$name <- Input_Data$new_name
	write.csv(sus_data, output_cox_predict, row.names = F)
	beta <- coef(m)
	se <- sqrt(diag(vcov(m)))
  	HR <- exp(beta)
  	HRse <- HR * se

  	#summary(m)
  	tmp <- round(cbind(coef = beta, se = se, z = beta/se, p = 1 - pchisq((beta/se)^2, 1),
                     HR = HR, HRse = HRse,
                     HRz = (HR - 1) / HRse, HRp = 1 - pchisq(((HR - 1)/HRse)^2, 1),
                     HRCILL = exp(beta - qnorm(.975, 0, 1) * se),
                     HRCIUL = exp(beta + qnorm(.975, 0, 1) * se)), 3)
	#print(tmp)
	write.csv(tmp, output_cox_path, row.names = T)
	print('++++++++++++++++++++++++++END++++++++++++++++++')
	if(svm_summary == 'Need'){
		sus_data = sus_data_n
		svm_sdf <- survivalsvm(Surv(OS, State) ~ ., data = sus_data, type = "regression", gamma.mu = 1, opt.meth = "quadprog", kernel = "add_kernel")
		print('++++++++++++++++++++++++++SVM+++++++++++++++++')
		pred.survsvm.reg <- predict(object = svm_sdf, newdata = sus_data)
		pred_svm <- as.vector(pred.survsvm.reg$predicted)
		one_median_pvalue <- ifelse(pred_svm > quantile(pred_svm)[3], 'High', 'Low')
		one_high_pvalue <- ifelse(pred_svm > quantile(pred_svm)[4], 'High', 'Low')
		one_low_pvalue <- ifelse(pred_svm > quantile(pred_svm)[2], 'High', 'Low')
		sus_data$median <- one_median_pvalue
		sus_data$low <- one_low_pvalue
		sus_data$high <- one_high_pvalue
		sus_data$pred_svm_value <- pred_svm

		high_sdf<-survdiff(formula = Surv(OS, State) ~ high, data = sus_data)
		low_sdf<-survdiff(formula = Surv(OS, State) ~ low, data = sus_data)
		median_sdf<-survdiff(formula = Surv(OS, State) ~ median, data = sus_data)
		high.p.val <- 1 - pchisq(high_sdf$chisq, length(high_sdf$n) - 1)
		print(high.p.val)
		low.p.val <- 1 - pchisq(low_sdf$chisq, length(low_sdf$n) - 1)
	        print(low.p.val)
		median.p.val <- 1 - pchisq(median_sdf$chisq, length(median_sdf$n) - 1)
	        print(median.p.val)
		sus_data$name <- Input_Data$new_name
		write.csv(sus_data, output_path, row.names = F)
		print('++++++++++++++++++++++++++END+++++++++++++++++')
	}
	return(sdf)
}
