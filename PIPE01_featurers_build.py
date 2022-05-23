import SimpleITK as sitk
import radiomics
import os
import logging
import sys
from radiomics import featureextractor, getFeatureClasses
import radiomics
import collections
import numpy
import numpy as np
import six
from radiomics import firstorder, glcm, imageoperations, shape, glrlm, glszm, gldm, ngtdm
import scipy.stats


def Run_Features_Find(image, mask, label, imageName, maskName):
	image_matrix = sitk.GetArrayFromImage(image)
	#print(image_m
	# cor signature calculation
	# These are currently set equal to the respective default values
	result_out = {}
	settings = {}
	settings['binWidth'] = 25
	#settings['resampledPixelSpacing'] = [1,1,1]
	settings['interpolator'] = sitk.sitkNearestNeighbor#'sitkBSpline'
	settings['verbose'] = True
	settings['normalize'] = True
	settings['normalizeScale'] = True
	settings['removeOutilers'] = True
	
	interpolator = settings.get('interpolator')
	resampledPixelSpacing = settings.get('resampledPixelSpacing')
	if interpolator is not None and resampledPixelSpacing is not None:
		image, mask = imageoperations.resampleImage(image, mask, **settings)

	bb, correctedMask = imageoperations.checkMask(image, mask, label=1)
	if correctedMask is not None:
		mask = correctedMask
	croppedImage, croppedMask = imageoperations.cropToTumorMask(image, mask, bb)

	#print('Calculating first order features...')
	firstOrderFeatures = firstorder.RadiomicsFirstOrder(croppedImage, croppedMask, **settings)
	firstOrderFeatures.enableFeatureByName('Mean', True)
	result = firstOrderFeatures.execute()
	for (key, val) in six.iteritems(result):
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating shape features...')
	shapeFeatures = shape.RadiomicsShape(croppedImage, croppedMask, **settings)
	shapeFeatures.enableAllFeatures()
	result = shapeFeatures.execute()
	for (key, val) in six.iteritems(result):
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLCM features...')
	glcmFeatures = glcm.RadiomicsGLCM(croppedImage, croppedMask, **settings)
	glcmFeatures.enableAllFeatures()
	result = glcmFeatures.execute()
	for (key, val) in six.iteritems(result):
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLRLM features...')
	glrlmFeatures = glrlm.RadiomicsGLRLM(croppedImage, croppedMask, **settings)
	glrlmFeatures.enableAllFeatures()
	result = glrlmFeatures.execute()
	for (key, val) in six.iteritems(result):
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLSZM features...')
	glszmFeatures = glszm.RadiomicsGLSZM(croppedImage, croppedMask, **settings)
	glszmFeatures.enableAllFeatures()
	result = glszmFeatures.execute()
	for (key, val) in six.iteritems(result):
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLDM features...')
	gldmFeatures = gldm.RadiomicsGLDM(croppedImage, croppedMask, **settings)
	gldmFeatures.enableAllFeatures()
	result = gldmFeatures.execute()
	for (key, val) in six.iteritems(result):
		result_out[key] = val

	#print('Calculating NGTDM features...')
	ngtdmFeatures = ngtdm.RadiomicsNGTDM(croppedImage, croppedMask, **settings)
	ngtdmFeatures.enableAllFeatures()
	result = ngtdmFeatures.execute()
	for (key, val) in six.iteritems(result):
		result_out[key] = val
	#print('Calculated Log filtered images:...')
	logFeatures = {}
	sigmaValues = [1.0, 3.0, 5.0]
	for logImage, imageTypename, inputSettings in imageoperations.getLoGImage(image, mask, sigma=sigmaValues):
		logImage, croppedMask = imageoperations.cropToTumorMask(logImage, mask, bb)
		logFirstorderFeatures = firstorder.RadiomicsFirstOrder(logImage, croppedMask, **inputSettings)
		logFirstorderFeatures.enableAllFeatures()
		logFeatures[imageTypename] = logFirstorderFeatures.execute()

	for sigma, features in six.iteritems(logFeatures):
		for (key, val) in six.iteritems(features):
			laplacianFeatureName = '%s_%s' % (str(sigma), key)
			result_out[laplacianFeatureName] = val
			#print('  ', laplacianFeatureName, ':', val)
	###### ADD Part One
	#print('Calculating shape features...')
	shapeFeatures = shape.RadiomicsShape(logImage, croppedMask, **settings)
	shapeFeatures.enableAllFeatures()
	result = shapeFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.logimage'
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLCM features...')
	glcmFeatures = glcm.RadiomicsGLCM(logImage, croppedMask, **settings)
	glcmFeatures.enableAllFeatures()
	result = glcmFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.logimage'
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLRLM features...')
	glrlmFeatures = glrlm.RadiomicsGLRLM(logImage, croppedMask, **settings)
	glrlmFeatures.enableAllFeatures()
	result = glrlmFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.logimage'
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLSZM features...')
	glszmFeatures = glszm.RadiomicsGLSZM(logImage, croppedMask, **settings)
	glszmFeatures.enableAllFeatures()
	result = glszmFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.logimage'
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLDM features...')
	gldmFeatures = gldm.RadiomicsGLDM(logImage, croppedMask, **settings)
	gldmFeatures.enableAllFeatures()
	result = gldmFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.logimage'
		result_out[key] = val

	#print('Calculating NGTDM features...')
	ngtdmFeatures = ngtdm.RadiomicsNGTDM(logImage, croppedMask, **settings)
	ngtdmFeatures.enableAllFeatures()
	result = ngtdmFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.logimage'
		result_out[key] = val





	#print('Calculated waveletFeatures:...')
	waveletFeatures = {}
	for decompositionImage, decompositionName, inputSettings in imageoperations.getWaveletImage(image, mask):
		decompositionImage, croppedMask = imageoperations.cropToTumorMask(decompositionImage, mask, bb)
		waveletFirstOrderFeaturs = firstorder.RadiomicsFirstOrder(decompositionImage, croppedMask, **inputSettings)
		waveletFirstOrderFeaturs.enableAllFeatures()
		waveletFeatures[decompositionName] = waveletFirstOrderFeaturs.execute()

	for decompositionName, features in six.iteritems(waveletFeatures):
		for (key, val) in six.iteritems(features):
			waveletFeatureName = '%s_%s' % (str(decompositionName), key)
			result_out[waveletFeatureName] = val
			#print('  ', waveletFeatureName, ':', val)
	### ADD Part
	###### ADD Part One
	#print('Calculating shape features...')
	shapeFeatures = shape.RadiomicsShape(decompositionImage, croppedMask, **settings)
	shapeFeatures.enableAllFeatures()
	result = shapeFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.decomposition.image'
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLCM features...')
	glcmFeatures = glcm.RadiomicsGLCM(decompositionImage, croppedMask, **settings)
	glcmFeatures.enableAllFeatures()
	result = glcmFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.decomposition.image'
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLRLM features...')
	glrlmFeatures = glrlm.RadiomicsGLRLM(decompositionImage, croppedMask, **settings)
	glrlmFeatures.enableAllFeatures()
	result = glrlmFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.decomposition.image'
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLSZM features...')
	glszmFeatures = glszm.RadiomicsGLSZM(decompositionImage, croppedMask, **settings)
	glszmFeatures.enableAllFeatures()
	result = glszmFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.decomposition.image'
		result_out[key] = val
		#print('  ', key, ':', val)

	#print('Calculating GLDM features...')
	gldmFeatures = gldm.RadiomicsGLDM(decompositionImage, croppedMask, **settings)
	gldmFeatures.enableAllFeatures()
	result = gldmFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.decomposition.image'
		result_out[key] = val

	#print('Calculating NGTDM features...')
	ngtdmFeatures = ngtdm.RadiomicsNGTDM(decompositionImage, croppedMask, **settings)
	ngtdmFeatures.enableAllFeatures()
	result = ngtdmFeatures.execute()
	for (key, val) in six.iteritems(result):
		key = key + '.decomposition.image'
		result_out[key] = val



	params = os.path.join('/home/zhangpeng/Project/Radiomics_Data/pyradiomics/examples/exampleSettings/Params.yaml')

	extractor = featureextractor.RadiomicsFeatureExtractor(params)
	extractor.enableFeatureClassByName('shape', enabled=False) 
	filters = {
	    "AdditiveGaussianNoise" : sitk.AdditiveGaussianNoiseImageFilter(),
	    "Bilateral" : sitk.BilateralImageFilter(),
	    "BinomialBlur" : sitk.BinomialBlurImageFilter(),
	    "BoxMean" : sitk.BoxMeanImageFilter(),
	    "BoxSigmaImageFilter" : sitk.BoxSigmaImageFilter(),
	    "CurvatureFlow" : sitk.CurvatureFlowImageFilter(),
	    "DiscreteGaussian" : sitk.DiscreteGaussianImageFilter(),
	    "LaplacianSharpening" : sitk.LaplacianSharpeningImageFilter(),
	    "Mean" : sitk.MeanImageFilter(),
	    "Median" : sitk.MedianImageFilter(),
	    "Normalize" : sitk.NormalizeImageFilter(),
	    "RecursiveGaussian" : sitk.RecursiveGaussianImageFilter(),
	    "ShotNoise" : sitk.ShotNoiseImageFilter(),
	    "SmoothingRecursiveGaussian" : sitk.SmoothingRecursiveGaussianImageFilter(),
	    "SpeckleNoise" : sitk.SpeckleNoiseImageFilter(),
	}
	results = {}
	results["baseline"] = extractor.execute(image, label)

	for key, value in six.iteritems(filters):
		#print ( "filtering with " + key )
		filtered_image = value.Execute(image)
		results[key] = extractor.execute(filtered_image, label)

	for (key, val) in six.iteritems(result):
		result_out[key] = val
		#print('  ', key, ':', val)
	params = os.path.join('/home/zhangpeng/Project/Radiomics_Data/pyradiomics/examples/exampleSettings/Params.yaml')
	extractor = featureextractor.RadiomicsFeatureExtractor(params)
	settings = {}
	settings['binWidth'] = 25
	settings['resampledPixelSpacing'] = None
	settings['interpolator'] = 'sitkBSpline'
	settings['verbose'] = True
	settings['binWidth'] = 25
	settings['resampledPixelSpacing'] = [3,3,3]
	settings['interpolator'] = sitk.sitkNearestNeighbor#'sitkBSpline'
	settings['verbose'] = True
	settings['normalize'] = True
	settings['normalizeScale'] = True
	settings['removeOutilers'] = True

	extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
	extractor.disableAllFeatures()
	extractor.enableFeatureClassByName('firstorder')
	featureVector = extractor.execute(imageName, maskName)
	featureVector = extractor.execute(image, mask)
	for featureName in featureVector.keys():
		result_out[featureName] = featureVector[featureName]
		#print('Computed %s: %s' % (featureName, featureVector[featureName]))
	return result_out

def sampletransfer(image,mask):
	rif = sitk.ResampleImageFilter()
	rif.SetReferenceImage(image)
	rif.SetOutputPixelType(mask.GetPixelID())
	rif.SetInterpolator(sitk.sitkNearestNeighbor)
	resMask = rif.Execute(mask)
	return resMask


def Run_two_group_data_to_txt(path_to_img, path_to_mask, path_to_img_adj, path_to_mask_adj,center_ID, Sample_ID, summary_ID):
	imageName, maskName = path_to_img, path_to_mask
	image = sitk.ReadImage(imageName)
	mask = sitk.ReadImage(maskName)
	#label = sitk.ReadImage(maskName)
	mask = sampletransfer(image, mask)
	input_image = sitk.Cast(image, sitk.sitkFloat32)
	mask_image = sitk.OtsuThreshold(input_image,0,1,200)
	corrector = sitk.N4BiasFieldCorrectionImageFilter()
	output_image = corrector.Execute(input_image,mask_image)
	image = sitk.Cast(output_image, sitk.sitkInt16)
	label = mask
	sitk.WriteImage(mask, path_to_mask_adj)
	sitk.WriteImage(image, path_to_img_adj)
	Out_result_Sample = Run_Features_Find(image, mask, label, path_to_img_adj, path_to_mask_adj)
	output_data_path = '/data/MRI/Out_Data_v3/' + center_ID + '_' +  Sample_ID + '_' +  summary_ID + '.txt'
	out_write_table = open(output_data_path, 'w')
	for one in Out_result_Sample.keys():
		one_str = str(one) + '\t' + str(Out_result_Sample[one]) + '\n'
		out_write_table.write(one_str)


with open('center1_file.txt','r') as Input_Data:
	All_Data_List = Input_Data.readlines()
	for One_Data_List in All_Data_List:
		One_Message = One_Data_List.rstrip().split('\t')
		print(One_Message)
		Orign_Path = One_Message[1] + '/' + One_Message[6]
		label_Path = One_Message[1] + '/' + One_Message[3]
		node_Path = One_Message[1] + '/' + One_Message[4]
		tumor_Path = One_Message[1] + '/' + One_Message[5]
		if One_Message[6]!='' and One_Message[3]!='':
			#print(One_Message[6])
			#print(One_Message[3])
			path_to_img = One_Message[1] + '/' + One_Message[6]
			path_to_mask = One_Message[1] + '/' + One_Message[3]
			path_to_img_adj = One_Message[1] + '/adj_' + One_Message[6]
			path_to_mask_adj = One_Message[1] + '/adj_' + One_Message[3]
			center_ID = 'center1'
			Sample_ID = One_Message[0]
			summary_ID = 'Label'
			try:
				Run_two_group_data_to_txt(path_to_img, path_to_mask, path_to_img_adj, path_to_mask_adj,center_ID, Sample_ID, summary_ID)
			except:
				print(Sample_ID)
		if One_Message[6]!='' and One_Message[4]!='':
			path_to_img = One_Message[1] + '/' + One_Message[6]
			path_to_mask = One_Message[1] + '/' + One_Message[4]
			path_to_img_adj = One_Message[1] + '/adj_' + One_Message[6]
			path_to_mask_adj = One_Message[1] + '/adj_' + One_Message[4]
			center_ID = 'center1'
			Sample_ID = One_Message[0]
			summary_ID = 'Node'
			try:
				Run_two_group_data_to_txt(path_to_img, path_to_mask, path_to_img_adj, path_to_mask_adj,center_ID, Sample_ID, summary_ID)
			except:
				print(Sample_ID)
		if One_Message[6]!='' and One_Message[5]!='':
			path_to_img = One_Message[1] + '/' + One_Message[6]
			path_to_mask = One_Message[1] + '/' + One_Message[5]
			path_to_img_adj = One_Message[1] + '/adj_' + One_Message[6]
			path_to_mask_adj = One_Message[1] + '/adj_' + One_Message[5]
			center_ID = 'center1'
			Sample_ID = One_Message[0]
			summary_ID = 'Tumor'
			try:
				Run_two_group_data_to_txt(path_to_img, path_to_mask, path_to_img_adj, path_to_mask_adj,center_ID, Sample_ID, summary_ID)
			except:
				print(Sample_ID)
		
	

