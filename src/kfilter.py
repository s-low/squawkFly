#!/usr/local/bin/python

import sys
import cv2
import cv2.cv as cv
import numpy as np


class KFilter(object):

    def __init__(self):

        self.kf = cv.CreateKalman(4, 2, 0)
        self.state = cv.CreateMat(4, 1, cv.CV_32FC1)
        self.proc_noise = cv.CreateMat(4, 1, cv.CV_32FC1)
        self.measurement = cv.CreateMat(2, 1, cv.CV_32FC1)

        # transition matrix init
        for j in range(4):
            for k in range(4):
                self.kf.transition_matrix[j, k] = 0
            self.kf.transition_matrix[j, j] = 1

        # | 1 0 1 0 | x  |   | x + vx |
        # | 0 1 0 1 | y  | = | y + vy |
        # | 0 0 1 0 | vx |   |   vx   |
        # | 0 0 0 1 | vy |   |   vy   |

        self.kf.transition_matrix[0, 2] = 1
        self.kf.transition_matrix[1, 3] = 1

        cv.SetIdentity(self.kf.measurement_matrix)

        # why these values.....
        processNoiseCovariance = 1e-4
        measurementNoiseCovariance = 1e-1
        errorCovariancePost = 0.1

        cv.SetIdentity(
            self.kf.process_noise_cov, cv.RealScalar(processNoiseCovariance))
        cv.SetIdentity(
            self.kf.measurement_noise_cov, cv.RealScalar(measurementNoiseCovariance))
        cv.SetIdentity(
            self.kf.error_cov_post, cv.RealScalar(errorCovariancePost))

        # predict location in f2 from state f1
        control_mat = cv.CreateMat(2, 1, cv.CV_32FC1)

        self.predicted = None
        self.corrected = None

    def getPreState(self):
        return self.kf.state_pre

    def getPostState(self):
        return self.kf.state_post

    def setPostState(self, x, y, vx, vy):
        self.kf.state_post[0, 0] = x
        self.kf.state_post[1, 0] = y
        self.kf.state_post[2, 0] = vx
        self.kf.state_post[3, 0] = vy

    def predict(self):
        self.predicted = cv.KalmanPredict(self.kf)

    def correct(self, x, y):
        self.measurement[0, 0] = x
        self.measurement[1, 0] = y
        # print "correcting against", x, y
        self.corrected = cv.KalmanCorrect(self.kf, self.measurement)

    def update(self, x, y):
        self.measurement[0, 0] = x
        self.measurement[1, 0] = y

        self.predicted = cv.KalmanPredict(self.kf)
        # print "correcting against", x, y
        self.corrected = cv.KalmanCorrect(self.kf, self.measurement)

    def getCorrected(self):
        return (self.corrected[0, 0], self.corrected[1, 0],
                self.corrected[2, 0], self.corrected[3, 0])

    def getPredicted(self):
        return (self.predicted[0, 0], self.predicted[1, 0],
                self.predicted[2, 0], self.predicted[3, 0])
