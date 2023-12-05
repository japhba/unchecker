import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import photutils
import scipy as sp
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pikepdf import Name, Pdf, PdfImage
from PIL import Image
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from scipy.stats import multivariate_normal
from tqdm import tqdm

# global default settings
downscale = 0.3
scaleFactor = 1
gamma = 2
maxpeaks = 100
gaussBG = 10.0
gaussMain = .2

# setup logger to show time
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline():
    def __init__(self, im):
        self.o_im = im.astype(np.float32)

        # set default parameters
        self.downscale = downscale
        self.scaleFactor = scaleFactor
        self.gamma = gamma
        self.maxpeaks = maxpeaks

        self.gaussBG = gaussBG
        self.gaussMain = gaussMain
        self.notch_size = 1

        self.load_settings()

    @property
    def settings(self):
        return {"downscale": self.downscale, "scaleFactor": self.scaleFactor, "maxpeaks": self.maxpeaks, "gamma": self.gamma, "gaussBG": self.gaussBG, "gaussMain": self.gaussMain, "notch_size": self.notch_size}

    def rescale(self, downscale=None):
        img = self.o_im
        if not downscale:
            downscale = self.downscale
        else:
            self.downscale = downscale

        self.im = cv2.resize(img, dsize=(int(img.shape[1] * downscale), int(img.shape[0] * downscale)),
                         interpolation=cv2.INTER_AREA)

        return self.im


    def fft(self):
        im = self.im
        logger.info("Begin fft", time.process_time())
        fft = sp.fft.fft2(im)

        r, p = np.abs(fft), np.angle(fft)
        self.p = p
        f_im_ = sp.fft.fftshift(r)

        self.f_im = f_im_

        return self.f_im

    def gaussFilter(self, scaleFactor=None, gaussMain=None, gaussBG=None):
        """
        Scales down the FFT spectrum after applying a Gauss filter to facilitate peakfinding.
        :param scaleFactor:
        :param gaussMain:
        :param gaussBG:
        :return:
        """

        if not scaleFactor:
            scaleFactor = self.scaleFactor
        else:
            scaleFactor = (scaleFactor)
            self.scaleFactor = scaleFactor
            
        if not gaussBG:
            gaussBG = self.gaussBG
        else:
            gaussBG = (gaussBG)
            self.gaussBG = gaussBG
            
        if not gaussMain:
            gaussMain = self.gaussMain
        else:
            gaussMain = (gaussMain)
            self.gaussMain = gaussMain

        im = self.f_im
        # denoise images
        logger.info("Begin gaussfiltering")
        # apply gauss filtering after log
        g_im = gaussian_filter(np.log(im), sigma=5 * downscale * gaussMain)
        g_im_bg = gaussian_filter(np.log(im), sigma=10 * downscale * gaussBG)

        x, y = g_im.shape
        g_im = cv2.resize(g_im, dsize=(int(y // scaleFactor), x // int(scaleFactor)), interpolation=cv2.INTER_AREA)
        g_im_bg = cv2.resize(g_im_bg, dsize=(int(y // scaleFactor), x // int(scaleFactor)), interpolation=cv2.INTER_AREA)


        self.g_im = g_im
        self.g_im_bg = g_im_bg

    def findPeaks(self, maxpeaks=None, notch_size=None, nsigma_trshd=0.05):
        
        if not maxpeaks:
            maxpeaks = self.maxpeaks
        else:
            maxpeaks = int(maxpeaks)
            self.maxpeaks = maxpeaks

        if notch_size:
            self.notch_size = notch_size

        scaleFactor = self.scaleFactor
        
        # choose coordinates
        def x2u(x, ax=None):
            j = 1 if not ax else ax
            i = 0 if not ax else ax
            return x / ((self.im.shape[i] + self.im.shape[j]) / 2)

        def u2x(u, ax=None):
            j = 1 if not ax else ax
            i = 0 if not ax else ax
            return int(u * ((self.im.shape[i] + self.im.shape[j]) / 2))

        xx2u = lambda x, ax=None: x2u(x * scaleFactor, ax)
        u2xx = lambda u, ax=None: u2x(u, ax) // scaleFactor

        # get the current, downsampled FFT spectrum
        g_im = self.g_im
        logger.info("Begin thrshd")
        thrshd = photutils.segmentation.detect_threshold(g_im, nsigma=nsigma_trshd, background=self.g_im_bg)

        # heavy function, therefore downsample image prior to peakfinding
        logger.info("Begin resizing maps")

        logger.info("Begin peakfinder")
        detected_peaks = photutils.detection.find_peaks(g_im, box_size=u2xx(0.04), threshold=thrshd, npeaks=maxpeaks)

        peaks = np.column_stack((detected_peaks["x_peak"][:], detected_peaks["y_peak"][:]))

        legit_peaks = []
        for point in peaks:
            x, y = point[0] * scaleFactor, point[1] * scaleFactor

            # exclude peaks that are close to the center DC mode of the image
            centerTolerance = 0.01
            if abs(x - self.im.shape[1] / 2) < u2x(centerTolerance, 1) and abs(y - self.im.shape[0] / 2) < u2x(centerTolerance, 0):
                continue

            legit_peaks.append((x, y))

        def notchMap(im, position, sigma):
            xx, yy = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
            x = np.stack((xx, yy), axis=-1)
            gauss = multivariate_normal.pdf(x, position, sigma)
            gauss = gauss / (np.max(gauss))

            return gauss

        # TODO not
        notch = np.sum([notchMap(self.f_im, peak, sigma=u2x(0.08*self.notch_size, 0)) for peak in legit_peaks[:]], axis=0)
        # clip close maxima
        notch = np.clip(notch, 0, 1)
        # invert
        notch = 1 - notch

        self.notch = notch
        return notch

    def backtransform(self):
        filtered = np.exp(self.notch * np.log(self.f_im))

        self.denotched_f = filtered

        # rebuild complex array
        r = filtered
        r = sp.fft.ifftshift(r)

        # backtransform
        f_im = r * np.exp(1j * self.p)
        im_filtered = sp.fft.ifft2(f_im)

        im = np.abs(im_filtered)

        self.im_filtered = im

        return im

    def post_process(self, gamma=None):

        if not gamma:
            gamma = self.gamma
        else:
            self.gamma = gamma

        im = self.im_filtered

        def normalize(im):
            # max out range
            im = np.clip(im, 0.0, 1)
            im -= np.min(im)
            im = (np.max(im) - np.min(im)) ** -1 * im
            # im -= np.min(im)
            return im

        def set_gamma(im, g):
            im = im ** g
            return im

        def clipShift(im, shift, n=True):
            # apply some clipping
            im -= shift
            im = np.clip(im, 0.0, 1)

            if n:
                im = normalize(im)

            return im

        # stack
        def stack(im):
            # increase contrast with parabola, choose exponent appropriately
            # im = im ** 2

            # increase blacks by normalizing
            # im = im / np.max(im)

            # normalize to unit interval
            im = im / 255.

            # # invert
            im = 1. - im

            # im = normalize(im)

            # # correct gamma
            # im = set_gamma(im, gamma)

            # # apply some clipping
            # im = clipShift(im, 0.08, n=1)

            # # correct gamma
            # im = set_gamma(im, gamma)

            # # apply some clipping
            # im = clipShift(im, 0.05, n=1)

            # # correct gamma
            # im = set_gamma(im, gamma)

            # # apply some clipping
            # im = clipShift(im, 0.35, n=1)

            # # backtransform
            im = 1. - im

            # undo the normalization
            im = im*255.

            return im

        im = stack(im)

        self.result = im
        return im

    def full_pipeline(self):
        self.rescale()
        self.fft()
        self.gaussFilter()
        self.findPeaks()
        self.backtransform()
        # self.postProcess()

    def save_settings(self):
        settings = self.settings
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json'), 'w') as f:
            json.dump(settings, f)

        logger.info("Settings saved!")

    def load_settings(self):
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json'), 'r') as f:
                settings = json.load(f)

            self.settings.update(settings)
            for k, v in settings.items():
                setattr(self, k, v)

            logger.info(f"Settings {settings} loaded!")

        except FileNotFoundError as e:
            logger.info("Settings from dry-run not found. Using default settings instead!")
            return
