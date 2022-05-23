import os


Filter_List = ['diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy', 'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet', 'diagnostics_Versions_Python', 'diagnostics_Configuration_Setting', 'diagnostics_Configuration_EnabledImageTypes']

def Run_Merge_table(center_ID, summary_ID):
	File_Data = '/home/zhangpeng/Project/Radiomics_Data/pyrad_data/NEW_Data/' + center_ID + '_file.txt'
	with open(File_Data, 'r') as Input_Data:
		Out_path_name = center_ID + '_' + summary_ID + '_out.txt'
		out_data_Data = open(Out_path_name , 'w')
		All_Data_List = Input_Data.readlines()
		k = 1
		for One_Data_List in All_Data_List:
			One_Message = One_Data_List.rstrip().split('\t')
			Orign_Path = One_Message[1] + '/' + One_Message[6]
			label_Path = One_Message[1] + '/' + One_Message[3]
			node_Path = One_Message[1] + '/' + One_Message[4]
			tumor_Path = One_Message[1] + '/' + One_Message[5]
			Sample_ID = One_Message[0]
			Data_path = '/data/MRI/Out_Data_v2/' + center_ID + '_' + Sample_ID + '_' + summary_ID + '.txt'
			if not os.path.isfile(Data_path):
				print(Data_path)
			if os.path.isfile(Data_path):
				with open(Data_path, 'r') as Part_Data:
					All_Part_Data = Part_Data.readlines()
					title_Message = 'ID'
					Out_Message = center_ID + '_' + Sample_ID + '_' + summary_ID
					#if k == 1:
					for One_Part_Data in All_Part_Data:
						MRI_One_Message = One_Part_Data.rstrip().split('\t')
						if 'diagnostics' not in MRI_One_Message[0]:
							#print(MRI_One_Message)
							title_Message = title_Message + '\t' + MRI_One_Message[0]
							Out_Message = Out_Message + '\t' + MRI_One_Message[1]
					title_Message = title_Message + '\n'
					Out_Message  = Out_Message + '\n'
					if k == 1:
						out_data_Data.write(title_Message)
						out_data_Data.write(Out_Message)
						k = 2
					else:
						out_data_Data.write(Out_Message)





