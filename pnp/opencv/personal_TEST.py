# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 19:57:55 2017

@author: Weiyan Cai
"""
from __future__ import division

import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# add path
import sys
sys.path.append(os.getcwd() + '/scripts')
import EPnP


class EPnPTest(object):
    def __init__(self):
        # self.load_test_data()
        self.epnp = EPnP.EPnP()

    def load_test_data(self):
        data = sio.loadmat(os.getcwd() + '/input/input_data_noise.mat')

        self.A = data['A']
        # self.Rt = data['Rt']      # not used (extrinsic matrix?? might be the gt extrinsic...)

        points = data['point']
        # self.n = len(points[0])   # not used
        self.Xcam, self.Ximg_true, self.Ximg_pix_true, self.Ximg, self.Ximg_pix, self.Xworld= [], [], [], [], [], []            # I think I need:
                                                                                                            # 1) camera coordinate frame points; used for verification
                                                                                                            # 2) 2D points in the image frame (what's seen/observed)
                                                                                                            # 3) 3D points in the world frame (ground truth)
        
        for p in points[0]:
            self.Xcam.append(p[0])                          # are these points in the camera coordinate frame?
            # self.Ximg_true.append(p[1])                   # not used
            # self.Ximg_pix_true.append(p[2])               # used as ground truth pixels
            # self.Ximg.append(p[3])                        # not used
            self.Ximg_pix.append(p[4])                      # 2D landmarks in image plane
            self.Xworld.append(p[5])                        # 3D actual points in world frame (according to )

    def load_custom_points(self, Xworld, Xcam, Ximg_pix, Ximg_pix_true, A):
        self.A = A                  # camera intrinsics (3x4)
        self.Xworld   = Xworld[..., np.newaxis]                 # 3D world coordinates
        self.Xcam     = Xcam[..., np.newaxis]                   # 3D camera coordinates (as a ground truth)
        self.Ximg_pix = Ximg_pix[..., np.newaxis]               # 2D image pixels OBSERVED
        self.Ximg_pix_true = Ximg_pix_true[..., np.newaxis]     # 2D image pixels if no transformation is applied (ground truth)
        # self.n = len(Ximg_pix)
    
    def load_custom_points_noplot(self, Xworld, Ximg_pix, A):
        self.A = A                  # camera intrinsics (3x4)
        self.Xworld   = Xworld[..., np.newaxis]                 # 3D world coordinates
        # self.Xcam     = Xcam[..., np.newaxis]                   # 3D camera coordinates (as a ground truth)
        self.Ximg_pix = Ximg_pix[..., np.newaxis]               # 2D image pixels OBSERVED
        # self.Ximg_pix_true = Ximg_pix_true[..., np.newaxis]     # 2D image pixels if no transformation is applied (ground truth)
        # self.n = len(Ximg_pix)

    def draw_input_noisy_data(self):                        # draws points of Ximg_pix_true and Ximg_pix on matplotlib
        fig = plt.figure()
        fig.set_size_inches(18.5, 13)
        axes = fig.add_subplot(1, 1, 1)
        plt.plot(0, 0, 'ok')
        for p in self.Ximg_pix_true:                        # gt pixels (not noisy)
            plt.plot(p[0], p[1], '.r')
        for p in self.Ximg_pix:                             # pixels in data (that are noisy)
            plt.plot(p[0], p[1], 'xg')
        axes.set_title('Noise in Image Plane', fontsize=18)
        plt.grid()
        
        fig.savefig(os.getcwd() + '/output/Noise_in_Image_Plane.png', dpi=100)
        plt.show()
        
    def apply_EPnP(self, plot:bool=True):
        error, Rt, Cc, Xc = self.epnp.efficient_pnp(np.array(self.Xworld), np.array(self.Ximg_pix), self.A)   # how to get self.Xworld, self.Ximg_pix, self.A: intrinsincs
        if plot: self.plot_3d_reconstruction("EPnP (Old)", Xc)
        # print("Error of EPnP: ", error)
        # print("Rt: ", Rt)
        # print("Cc: ", Cc)
        return error, Rt, Cc, Xc
        
    def apply_EPnP_Gauss(self, plot:bool=True):
        error, Rt, Cc, Xc = self.epnp.efficient_pnp_gauss(np.array(self.Xworld), np.array(self.Ximg_pix), self.A)
        if plot: self.plot_3d_reconstruction("EPnP (Gauss Newton)", Xc)
        # print("Error of EPnP (Gauss Newton Optimization): ", error)
        # print("Rt: ", Rt)
        # print("Cc: ", Cc)
        return error, Rt, Cc, Xc
        
    def plot_3d_reconstruction(self, method, Xc):
        fig = plt.figure()
        fig.set_size_inches(18.5, 13)
        axes = fig.add_subplot(1, 1, 1)
        plt.plot(0, 0, 'ok')
        for p in self.Xcam:                     # 3d camera frame coordinates?
            plt.plot(p[0], p[1], '.r')
        # for p in self.Xworld:
            # plt.plot(p[0], p[1], '.r')
        for p in Xc:                            # I think 3d coordinate estimated by EPnP???
            plt.plot(p[0], p[1], 'xg')
        axes.set_title(method + ' - Reprojection Error', fontsize=18)
        plt.grid()
        
        fig.savefig(os.getcwd() + "/output/" + method + '_Reprojection_Error.png', dpi=100)
        plt.show()
    

if __name__ == "__main__":
    ET = EPnPTest()
    ET.load_test_data()
    ET.draw_input_noisy_data()
    ET.apply_EPnP()
    ET.apply_EPnP_Gauss()