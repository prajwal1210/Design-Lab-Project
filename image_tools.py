# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt
import copy	


class ImageAnalyser():
    def __init__(self,image_path):
        self.image = cv2.imread(image_path)
        self.mask = None

    def preprocessImage(self, image):
        preprocessedImage = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
        preprocessedImage = cv2.cvtColor(preprocessedImage, cv2.COLOR_BGR2GRAY)
        preprocessedImage = cv2.GaussianBlur(preprocessedImage, (11, 11), 0)
        return preprocessedImage
    
    def binarizeImage(self,image):
        binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]

        #Erosion and dilation
        binary_image = cv2.erode(binary_image, None, iterations=2)
        binary_image = cv2.dilate(binary_image, None, iterations=4)
        return binary_image
    
    def calculateMask(self):
        preprocessedImage  = self.preprocessImage(self.image)
        binary_image = self.binarizeImage(preprocessedImage)
        
        labels = measure.label(binary_image, neighbors=8, background=0)
        self.mask = np.zeros(binary_image.shape, dtype="uint8")

        max_region_avg = 0
        numPixelmax = 0
        for label in np.unique(labels):
            #Background
            if label == 0:
                continue
            else:
                label_mask = np.zeros(binary_image.shape,dtype='uint8')
                label_mask[labels == label] = 1

                #Erode and dilate the mask
                label_mask = cv2.erode(label_mask, None, iterations=20)
                label_mask = cv2.dilate(label_mask, None, iterations=20)

                #Extract the region and find the average intensity
                region_image = np.multiply(preprocessedImage,label_mask)
                numPixels = cv2.countNonZero(region_image)
                
                #If number of pixels in this region is less than a certain threshold, continue
                if numPixels < 3000:
                    continue

                #See if the region's avg intensity is the max
                
                region_avg = 0
                for row in range(region_image.shape[0]):
                    for col in range(region_image.shape[1]):
                        region_avg+=region_image[row][col]
                region_avg = region_avg/numPixels
                
                if region_avg > max_region_avg+2:
                    max_region_avg = region_avg
                    numPixelmax = numPixels
                    self.mask = label_mask
                elif region_avg > max_region_avg and region_avg < max_region_avg+2:
                    if numPixels > numPixelmax:
                        max_region_avg = region_avg
                        numPixelmax = numPixels
                        self.mask = label_mask
    
    def getRegionImage(self):
        if self.mask == None:
            self.calculateMask()
        preprocessedImage = self.preprocessImage(self.image)
        region_image = np.multiply(self.mask,preprocessedImage)
        return region_image
    
    def getEqualizedRegionImage(self):
        region_image = self.getRegionImage()
        region_image = cv2.equalizeHist(region_image)
        return region_image

    def getPreprocessedImage(self):
        return self.preprocessImage(self.image)
    
    def getBlur(self):
        return cv2.Laplacian(self.image, cv2.CV_64F).var()

class Detector(ImageAnalyser):

    def calculateDomeMask(self):
        preprocessedImage  = self.preprocessImage(self.image)
        binary_image = self.binarizeImage(preprocessedImage)
        
        labels = measure.label(binary_image, neighbors=8, background=0)
        self.mask = np.zeros(binary_image.shape, dtype="uint8")
        
        masks = []
        
        for label in np.unique(labels):
            #Background
            if label == 0:
                continue
            else:
                label_mask = np.zeros(binary_image.shape,dtype='uint8')
                label_mask[labels == label] = 1

                #Erode and dilate the mask
                label_mask = cv2.erode(label_mask, None, iterations=30)
                label_mask = cv2.dilate(label_mask, None, iterations=30)
                
                #Extract the region and find the average intensity
                region_image = np.multiply(preprocessedImage,label_mask)
                numPixels = cv2.countNonZero(region_image)
                
                #If number of pixels in this region is less than a certain threshold, continue
                if numPixels < 20000:
                    continue
                
                cnts = cv2.findContours(label_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                if len(cnts) !=0:
                    cnts = contours.sort_contours(cnts)[0]
                    cnt_list = []
                    for i, c in enumerate(cnts):
                        # draw the bright spot on the image
                        (x, y, w, h) = cv2.boundingRect(c)
                        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
                        cnt_list.append(((cX, cY), radius))

                    sorted(cnt_list,key = lambda x: x[1],reverse=True)
                    ((m_cX, m_cY), m_radius) = cnt_list[0]
                    total = 0
                    count = 0
                    for row in range(label_mask.shape[0]):
                        for col in range(label_mask.shape[1]):
                            v = (col-m_cX)**2 + (row-m_cY)**2
                            if v < m_radius**2 :
                                if label_mask[row][col]!=0:
                                    count += 1
                                total+=1
                    if (float(count)/total) <= 0.1:
                        continue
                
                region_avg = 0
                for row in range(region_image.shape[0]):
                    for col in range(region_image.shape[1]):
                        region_avg+=region_image[row][col]
                region_avg = region_avg/numPixels
                masks.append((region_avg,numPixels,label_mask))
        
        sorted(masks,reverse=True)
        if len(masks) > 3:
            masks = masks[:3]
        
        #Filter out bad masks
        final_masks = []
        if len(masks)!=0:
            maxRegionInt = masks[0][0]

            sorted(masks,key = lambda x: x[1],reverse=True)

            maxNumPixels = masks[0][1]

            #Based on the size of the masks filter them out
            for mask in masks:
                if(mask[0]/maxRegionInt >= 0.7 and mask[1]/maxNumPixels >= 0.6):
                    final_masks.append(mask)

        
        for mask in final_masks:
            self.mask = cv2.add(self.mask,mask[2])
        
        return (self.mask,len(final_masks))
          
    def getDomeRegionImage(self):
        if self.mask == None:
            self.calculateDomeMask()
        preprocessedImage = self.preprocessImage(self.image)
        region_image = np.multiply(self.mask,preprocessedImage)
        return region_image
