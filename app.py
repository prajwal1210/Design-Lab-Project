from imutils import contours
import numpy as np
import argparse 
import imutils
import cv2
from matplotlib import pyplot as plt
import glob
from image_tools import ImageAnalyser, Detector
import re
import os
import shutil
from PIL import Image, ExifTags
import flask
from flask import Flask,request,jsonify, make_response
import logging
from joblib import Parallel, delayed


app = Flask(__name__) 

app.config['LUMENS_THRESHOLD'] = 0.2 # % of Reference Lumens that is maximum allowed deviation
app.config['BLUR_THRESHOLD'] = 0.2 # % of Blur Value that is maximum allowed deviation
app.logger.setLevel(logging.INFO)

# Calculates the blur of the image using the ImageAnalyzer class via the Variance of Gaussian method
def calcBlur(image):
    analzer_obj = ImageAnalyser(image)
    blur = analzer_obj.getBlur()
    return blur

# Calculates the Lumens for the image using the ImageAnalyzer Class
def calcLumensForImage(image):
    analzer_obj = ImageAnalyser(image)

    # Find the average illumination
    avg_intensity = 0
    preprocessedImage = analzer_obj.getPreprocessedImage()
    for i in range(preprocessedImage.shape[0]):
        for j in range(preprocessedImage.shape[1]):
            avg_intensity += preprocessedImage[i][j]

    avg_intensity = float(avg_intensity)/(preprocessedImage.shape[0]*preprocessedImage.shape[1])

    img = Image.open(image)
        
    # Get Metadata
    exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    
    # Use the metadata to calculate the lumens detected from the average intensity
    try:
        iso_rating = exif['ISOSpeedRatings']
        fnumber = float(exif['FNumber'][0])/exif['FNumber'][1]
        exp_time = float(exif['ExposureTime'][0])/exif['ExposureTime'][1]
        
        
        lumens = (avg_intensity*(fnumber**2))/(iso_rating*exp_time)
    except:
        app.logger.info('Could not calculate the lumens due to lack of metadata')
        lumens = avg_intensity

    return lumens

# Wrapper function that calculates both Lumens and Blur for the input image by calling the appropriate functions
def calcAll(image):
    return [calcLumensForImage(image),calcBlur(image)]

# A general comparator function that compares two value upto a certain threshold of tolerance. Note that this is is a lower threshold, so if curr_val > ref_val then the answer is true always 
def compareValues(curr_val,ref_val,threshold):
    allowed_deviation = threshold*ref_val
    if curr_val < (ref_val-allowed_deviation):
        return False
    else:
        return True

# Calculates lumes for the given image and compares it with a reference lumen value
def compareWithReferenceLumens(image, ref_lumens, lumens_threshold):
    curr_lumens = calcLumensForImage(image)
    return (compareValues(curr_lumens,ref_lumens,lumens_threshold),curr_lumens)

# Calculates reference lumes from the reference image and the calls compareWithReferenceLumens to compare the given image
def compareWithReference(image, ref_image, lumens_threshold):
    ref_lumens = calcLumensForImage(ref_image)
    return compareWithReferenceLumens(image,ref_lumens,lumens_threshold)

# Returns whether the image is blured or not
def isBlurred(image, ref_blur, blur_threshold):
    image_blur = calcBlur(image)
    return (compareValues(image_blur,ref_blur,blur_threshold),image_blur)

## Endpoint to calculate several things for a single current image (pair of one current image and one reference image)
# Request parameters: Current Image Path, Reference Blur Value(optional), Reference Image Path(optional)
    # Request Data Fields - 
        # “Curr_image” - Path to the currently uploaded image/image for which we want to run the query
        # “Reference_image” - Path to a reference image with which we compare the lumens of the given image
        # “Reference_Blur” - Reference blur value to compare with the given image’s blur value and see if it needs to be filtered out or not 
    # Response Fields -
        # “DetectedImageLumens”: Value of the detected lumens
        # “BlurValue”: Blur Value for the image
        # “IsBlurred”: Is the image blurred or not based on the reference blur value (only when reference blur was passed to it)
        # “IsQualityGood”: Is the quality of the light good when comparing lumens detected with the reference image
##
@app.route('/',methods=['POST'])
def serve_request():
    try:
        if request.method == 'POST':
            curr_image = request.json['current_image']
            resp = {}
            # If reference blur is provided, then compare and judge whether the image is blurred or not
            if 'reference_blur' in request.json:
                ref_blur = request.json['reference_blur']
                is_blurred, blur_val = isBlurred(curr_image,ref_blur,app.config['BLUR_THRESHOLD'])
                resp['IsBlurred'] = is_blurred
                resp['BlurValue'] = blur_val
            else:
                blur_val = calcBlur(curr_image)
                resp['BlurValue'] = blur_val
            # If the reference 
            if 'reference_image' in request.json:
                ref_image = request.json['reference_image']
                is_quality_good,curr_lumens = compareWithReference(curr_image,ref_image,app.config['LUMENS_THRESHOLD'])
                resp['IsQualityGood'] = is_quality_good
                resp['DetectedImageLumens'] = curr_lumens
            else:
                curr_lumens = calcLumensForImage(curr_image)
                resp['DetectedImageLumens'] = curr_lumens

            return make_response(jsonify(resp),200)
    except Exception as e:
        app.logger.info('Error Occured while serving post request')
        return make_response(jsonify(message='An internal error occured', error=str(e)), 500) 

## Main Endpoint to compare set of reference images with a set of the currently taken images
# Parameters: Set of paths of current images, Set of paths of reference images
    # Request data fields :
        # “Curr_images” - List of paths to the images in the current set of images we want to query for
        # “Reference_images” - List of paths to the images in the reference image set
    # Response fields :
        # “Images”: A dictionary with the fields:
            # ImagePath of the image in the current set (for each image one key): Dictionary with the fields:
                # “DetectedImageLumens”:  Value of the detected lumens
                # “BlurValue”: Blur Value for the image
                # “IsBlurred”: Is the image blurred or not based on the reference blur value (only when reference blur was passed to it)
        #“AllBlurred” :  Whether all images were considered blurred with respect to the reference(In case this is true, the next two fields are not present in the response)
        # “AverageLumensDetected” : Average of the lumens detected for each image
        # “IsQualityGood”: Is the quality of the light good when comparing average lumens detected with the reference set of images
        # “AverageReferenceLumens” : The average reference lumens as calculated from the set of reference images
        # “AverageReferenceBlur” : The average reference blur as calculated from the set of reference images
##
@app.route('/serveMultiRequest',methods=['POST'])
def  serve_multi_req():
    try:
        if request.method == 'POST':
            curr_images = request.json['curr_images']
            reference_images = request.json['reference_images']
            # Parallely compute the blur and lumens for both the image and references 
            reference_list = Parallel(n_jobs=5,verbose=10)(delayed(calcAll)(image) for image in reference_images)
            curr_list = Parallel(n_jobs=5,verbose=10)(delayed(calcAll)(image) for image in curr_images)
            resp = {}
            reference_lumens = 0
            reference_blur = 0

            for ref_image in reference_list:
                reference_lumens += ref_image[0]
                reference_blur += ref_image[1]
            
            reference_lumens = float(reference_lumens)/len(reference_list)
            reference_blur = float(reference_blur)/len(reference_list)

            resp['AverageReferenceLumens'] = reference_lumens
            resp['AverageReferenceBlur'] = reference_blur

            resp["Images"] = {}
            curr_lumens = 0
            curr_count = 0
            for image, curr_vals in zip(curr_images,curr_list):
                resp["Images"][image] = {}
                resp["Images"][image]["BlurValue"] = curr_vals[1]
                resp["Images"][image]["DetectedImageLumens"] = curr_vals[0]

                is_blurred = compareValues(resp["Images"][image]["BlurValue"],reference_blur,app.config['BLUR_THRESHOLD'])
                resp["Images"][image]["IsBlurred"] = is_blurred

                if not is_blurred:
                    curr_count+=1
                    curr_lumens += resp["Images"][image]["DetectedImageLumens"]
            
            if curr_count!=0:
                resp["AllBlurred"] = False
                curr_lumens = float(curr_lumens)/curr_count
                resp["AverageLumensDetected"] = curr_lumens
                resp["IsQualityGood"] = compareValues(curr_lumens,reference_lumens,app.config['LUMENS_THRESHOLD'])
            else: # If all images are blurred, we just output AllBlurred as true
                resp["AllBlurred"] = True

            return make_response(jsonify(resp),200)
    
    except Exception as e:
        app.logger.info('Error Occured while serving post request')
        return make_response(jsonify(message='An internal error occured', error=str(e)), 500)

## Endpoint where image to query is sent and we get the prediction as to how many LEDs are working
    # Request data fields :
        # “image ” - path to the query
    # Response fields :
        # “AllWorking” : Two class classification output whether all LEDs are working or not 
        # “DetectedLEDs” : Number of working LEDs Detected
##
@app.route('/detect',methods=['POST'])
def detect():
    try:
        if request.method == 'POST':
            image = request.json['image'] # Image Path
            detector = Detector(image)
            _,regions = detector.calculateDomeMask()
            resp = {}
            resp['DetectedLEDs'] = regions
            if regions < 3:
                resp['AllWorking'] = False
            else:
                resp['AllWorking'] = True

            return make_response(jsonify(resp),200)
    
    except Exception as e:
        app.logger.info('Error Occured while serving post request')
        return make_response(jsonify(message='An internal error occured', error=str(e)), 500)



if __name__ == "__main__":
        app.run()
