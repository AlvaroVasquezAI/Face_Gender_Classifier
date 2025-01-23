import cv2
import numpy as np
import skimage
import torch
import torch.nn as nn
from tqdm import tqdm

class Image:
    def __init__(self, image):
        self.image = self.preprocessImage(image)
        #Basics features
        self.colorChannelsRGB = self.extractColorChannelsRGB()
        self.RGBMean = self.calculateRGBMean()
        self.RGBMode = self.calculateRGBMode()
        self.RGBVariance = self.calculateRGBVariance()
        self.RGBStandardDeviation = self.calculateRGBStandardDeviation()
        self.colorHistogram = self.calculateColorHistogram()
        #Advanced features
        self.grayLevelCooccurrenceMatrixProperties = self.calculateGrayLevelCooccurrenceMatrixProperties()
        self.histogramOfOrientedGradients = self.calculateHistogramOfOrientedGradients()
        self.peakLocalMax = self.calculatePeakLocalMax()
        self.huMoments = self.getHuMoments()
        self.edgeDensity = self.calculateEdgeDensity()
        self.imageEntropy = self.calculateImageEntropy()
        self.laplacianMeanStd = self.calculateLaplacianMeanStd()
        self.aspectRatio = self.calculateAspectRatio()
        self.circularity = self.calculateCircularity()
        #Feature vector
        self.featureVector = self.generateFeatureVector()

    def preprocessImage(self, image):
        image = cv2.resize(image, (128, 128))
        return image

    def extractColorChannelsRGB(self):
        redChannel = self.image[:,:,0]
        greenChannel = self.image[:,:,1]
        blueChannel = self.image[:,:,2]

        return [redChannel, greenChannel, blueChannel]

    def calculateRGBMean(self):
        redChannel = self.image[:,:,0]
        greenChannel = self.image[:,:,1]
        blueChannel = self.image[:,:,2]
        redMean = np.mean(redChannel)
        greenMean = np.mean(greenChannel)
        blueMean = np.mean(blueChannel)

        return [redMean, greenMean, blueMean]

    def calculateRGBMode(self):
        redChannel = self.image[:,:,0]
        greenChannel = self.image[:,:,1]
        blueChannel = self.image[:,:,2]
        redMode = skimage.exposure.histogram(redChannel)[1].argmax()
        greenMode = skimage.exposure.histogram(greenChannel)[1].argmax()
        blueMode = skimage.exposure.histogram(blueChannel)[1].argmax()

        return [redMode, greenMode, blueMode]

    def calculateRGBVariance(self):
        redChannel = self.image[:,:,0]
        greenChannel = self.image[:,:,1]
        blueChannel = self.image[:,:,2]
        redVariance = np.var(redChannel.flatten())
        greenVariance = np.var(greenChannel.flatten())
        blueVariance = np.var(blueChannel.flatten())

        return [redVariance, greenVariance, blueVariance]

    def calculateRGBStandardDeviation(self):
        redChannel = self.image[:,:,0]
        greenChannel = self.image[:,:,1]
        blueChannel = self.image[:,:,2]
        redStandardDeviation = np.std(redChannel)
        greenStandardDeviation = np.std(greenChannel)
        blueStandardDeviation = np.std(blueChannel)

        return [redStandardDeviation, greenStandardDeviation, blueStandardDeviation]

    def calculateColorHistogram(self):
        image = self.image
        bins = 256
        # Calculate the histogram for each channel
        histogram = [cv2.calcHist([image], [i], None, [bins], [0, 256]) for i in range(3)]
        # Normalize the histograms
        histogram = [cv2.normalize(hist, hist).flatten() for hist in histogram]

        return histogram

    def calculateGrayLevelCooccurrenceMatrixProperties(self):
        image_gray = skimage.color.rgb2gray(self.image)
        image_gray_u8 = (image_gray * 255).astype(np.uint8)
        glcm = skimage.feature.graycomatrix(image_gray_u8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = skimage.feature.graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = skimage.feature.graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0]
        energy = skimage.feature.graycoprops(glcm, 'energy')[0, 0]
        correlation = skimage.feature.graycoprops(glcm, 'correlation')[0, 0]

        return [contrast, dissimilarity, homogeneity, energy, correlation]

    def calculateHistogramOfOrientedGradients(self):
        image_gray = skimage.color.rgb2gray(self.image)

        return skimage.feature.hog(image_gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), orientations=9, visualize=False)

    def calculatePeakLocalMax(self):
        image_gray = skimage.color.rgb2gray(self.image)

        return skimage.feature.peak_local_max(image_gray, min_distance=1, threshold_abs=0.1, num_peaks=10)

    def getHuMoments(self):
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
        _, image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        moments = cv2.moments(image_gray)
        huMoments = cv2.HuMoments(moments)

        return huMoments

    def calculateEdgeDensity(self):
        edges = cv2.Canny(self.image, 100, 200)
        edge_density = np.sum(edges > 0) / (self.image.shape[0] * self.image.shape[1])
        return edge_density

    def calculateImageEntropy(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(gray_image, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def calculateLaplacianMeanStd(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return [np.mean(laplacian), np.std(laplacian)]

    def calculateAspectRatio(self):
        h, w, _ = self.image.shape
        return w / h

    def calculateCircularity(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                return circularity
        return 0.0

    def generateFeatureVector(self):
        featureVector = np.array([])

        featureVector = np.append(featureVector, self.RGBMean)
        featureVector = np.append(featureVector, self.RGBMode)
        featureVector = np.append(featureVector, self.RGBVariance)
        featureVector = np.append(featureVector, self.RGBStandardDeviation)
        featureVector = np.append(featureVector, np.concatenate([ histogram.flatten() for histogram in self.colorHistogram ]))
        featureVector = np.append(featureVector, self.grayLevelCooccurrenceMatrixProperties)
        featureVector = np.append(featureVector, self.histogramOfOrientedGradients)
        featureVector = np.append(featureVector, self.peakLocalMax)
        featureVector = np.append(featureVector, self.huMoments)
        featureVector = np.append(featureVector, self.edgeDensity)
        featureVector = np.append(featureVector, self.imageEntropy)
        featureVector = np.append(featureVector, self.laplacianMeanStd)
        featureVector = np.append(featureVector, self.aspectRatio)
        featureVector = np.append(featureVector, self.circularity)

        return featureVector
    
class Perceptron(nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.history = {'loss': [], 'accuracy': []}

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

    def fit(self, X_train, y_train, max_iter, tol, criterion, optimizer):
        self.train()
        prev_loss = float('inf')

        for epoch in range(max_iter):
            optimizer.zero_grad()
            outputs = self(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            predictions = (outputs >= 0.5).float()
            accuracy = (predictions == y_train).float().mean()

            self.history['loss'].append(loss.item())
            self.history['accuracy'].append(accuracy.item())

            if abs(prev_loss - loss.item()) < tol:
                break
            prev_loss = loss.item()

    def evaluate(self, X_test):
        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            predictions = (outputs >= 0.5).float()
        return predictions.cpu().numpy()

    def predict(self, X_new):
        self.eval()
        with torch.no_grad():
            outputs = self(X_new)
            predictions = (outputs >= 0.5).float()
        return predictions.cpu().numpy()
