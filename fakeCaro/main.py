import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import multivariate_normal
import photutils
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import scipy as sp
import time
import sys
from tqdm import tqdm
import img2pdf
import tempfile
import os
from PIL import Image

from pikepdf import Pdf, PdfImage, Name

import argparse

from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib
matplotlib.use('Qt5Agg')


try:
    parser = argparse.ArgumentParser(description='An fft-based grid removal filter')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--downscale', '-d', default=92, type=int,
                        help='Threshold level for normalized magnitude spectrum')
    parser.add_argument('--scaleFactor', '-r', default=6, type=int,
                        help='Radius to expand the area of mask pixels')
    parser.add_argument('--maxpeaks', '-m', default=4, type=int,
                        help='Ratio for middle preservation')
    parser.add_argument('--gamma', '-m', default=1.0, type=float,
                        help='Ratio for middle preservation')

    parser.add_argument('--dry_run', '-d', default=True, type=bool,
                        help='Ratio for middle preservation')

    args = parser.parse_args()
    input_fn = args.input
    output_fn = args.ouput
    downscale = args.downscale
    scaleFactor = args.scaleFactor
    gamma = args.gamma
    maxpeaks = args.maxpeaks
    debug = args.debug
    dry_run = args.dry_run

except:
    # global settings
    downscale = 0.5
    scaleFactor = 3
    gamma = 2
    maxpeaks = 300
    dry_run = True
    input_fn = "script_qtcm_1s.pdf"

from pikepdf import Pdf, PdfImage, Name
import zlib

def pdfToImgs(filename, tmpdir=None):
    pdf = Pdf.open(filename)

    page = pdf.pages[0]
    image_name, rawimage = next(page.images.items())

    imgs = []

    for i, page in enumerate(pdf.pages):
        image_name, rawimage = next(page.images.items())

        pdfimage = PdfImage(rawimage)

        rawimage = pdfimage.obj
        pilImage = pdfimage.as_pil_image().convert("L")
        img = np.array(pilImage)
        img = cv2.resize(img, dsize=(int(img.shape[1] * downscale), int(img.shape[0] * downscale)),
                         interpolation=cv2.INTER_CUBIC)

        if tmpdir:
            Image.fromarray(img).save(os.path.join(tmpdir, "pre", f"{i}.png"))

        imgs.append(img)

    return imgs


def imgsToPdf(imgs, filename, tmpdir):
    imgs_fns = []
    for root, dirs, files in os.walk(os.path.join(tmpdir, "post")):
        for file in files:
            imgs_fns.append(os.path.join(root, file))

    # multiple inputs (variant 2)
    with open(filename, "wb") as f:
        f.write(img2pdf.convert(imgs_fns))


class Pipeline():
    def __init__(self, im):
        self.o_im = im.astype(np.float32)

        # set default parameters
        self.downscale = downscale
        self.scaleFactor = scaleFactor
        self.gamma = gamma
        self.maxpeaks = maxpeaks

        self.gaussBG = 1.0
        self.gaussMain = 1.0

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
        print("Begin fft", time.process_time())
        fft = sp.fft.fft2(im)

        r, p = np.abs(fft), np.angle(fft)
        self.p = p
        f_im_ = sp.fft.fftshift(r)

        self.f_im = f_im_

        return self.f_im

    def gaussFilter(self, scaleFactor=None, gaussMain=None, gaussBG=None):

        if not scaleFactor:
            scaleFactor = self.scaleFactor
        else:
            scaleFactor = int(scaleFactor)
            self.scaleFactor = scaleFactor
            
        if not gaussBG:
            gaussBG = self.gaussBG
        else:
            gaussBG = int(gaussBG)
            self.gaussBG = gaussBG
            
        if not gaussMain:
            gaussMain = self.gaussMain
        else:
            gaussMain = int(gaussMain)
            self.gaussMain = gaussMain

        im = self.f_im
        # denoise images
        print("Begin gaussfiltering", time.process_time())
        # apply gauss filtering after log
        g_im = gaussian_filter(np.log(im), sigma=5 * downscale * gaussMain)
        g_im_bg = gaussian_filter(np.log(im), sigma=10 * downscale * gaussBG)

        x, y = g_im.shape
        g_im = cv2.resize(g_im, dsize=(y // scaleFactor, x // scaleFactor), interpolation=cv2.INTER_AREA)
        g_im_bg = cv2.resize(g_im_bg, dsize=(y // scaleFactor, x // scaleFactor), interpolation=cv2.INTER_AREA)


        self.g_im = g_im
        self.g_im_bg = g_im_bg

    def findPeaks(self, maxpeaks=None, nsigma_trshd=0.05):
        
        if not maxpeaks:
            maxpeaks = self.maxpeaks
        else:
            maxpeaks = int(maxpeaks)
            self.maxpeaks = maxpeaks
        
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
        g_im = self.g_im
        print("Begin thrshd", time.process_time())
        thrshd = photutils.detection.detect_threshold(g_im, nsigma=nsigma_trshd, background=self.g_im_bg)

        # heavy function, therefore downsample image prior to peakfinding
        print("Begin resizing maps", time.process_time())

        print("Begin peakfinder", time.process_time())
        detected_peaks = photutils.detection.find_peaks(g_im, box_size=u2xx(0.05), threshold=thrshd, npeaks=maxpeaks)

        peaks = np.column_stack((detected_peaks["x_peak"][:], detected_peaks["y_peak"][:]))

        legit_peaks = []
        for point in peaks:
            x, y = point[0] * scaleFactor, point[1] * scaleFactor

            # exclude peaks that are close to the center DC mode of the image
            if abs(x - self.im.shape[1] / 2) < u2x(0.05, 1) and abs(y - self.im.shape[0] / 2) < u2x(0.05, 0):
                continue

            legit_peaks.append((x, y))

        def notchMap(im, position, sigma):
            xx, yy = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
            x = np.stack((xx, yy), axis=-1)
            gauss = multivariate_normal.pdf(x, position, sigma)
            gauss = gauss / np.max(gauss)

            return gauss

        notch = np.sum([notchMap(self.f_im, peak, sigma=u2x(0.03, 0)) for peak in legit_peaks[:]], axis=0)
        # clip close maxima
        notch = np.clip(notch, 0, 1)
        # invert
        notch = 1 - notch

        self.notch = notch
        return notch

    def backtransform(self):
        filtered = self.notch * self.f_im

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

    def postProcess(self, gamma=None):

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
            im = im ** 2

            # increase blacks by normalizing
            im = im / np.max(im)

            # normalize
            im = im / 255.0

            # invert
            im = 1 - im

            im = normalize(im)

            # correct gamma
            im = set_gamma(im, gamma)

            # apply some clipping
            im = clipShift(im, 0.08, n=1)

            # correct gamma
            im = set_gamma(im, gamma)

            # apply some clipping
            im = clipShift(im, 0.05, n=1)

            # correct gamma
            im = set_gamma(im, gamma)

            # apply some clipping
            im = clipShift(im, 0.35, n=1)

            # backtransform
            im = 1 - im

            # im = im*255

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
        self.postProcess()


def batchFilter(tmpdir):
    _, _, imgs = next(os.walk(os.path.join(tmpdir, "pre")))

    for i, img_fn in tqdm(enumerate(imgs)):
        img = Image.open(os.path.join(tmpdir, "pre", img_fn)).convert("L")
        img = np.array(img)

        p = Pipeline(img)
        p.rescale()
        p.fft()
        p.gaussFilter()
        p.findPeaks()
        p.backtransform()
        p.postProcess()

        out_im = Image.fromarray(p.result).convert("L")
        out_im.save(os.path.join(tmpdir, "post", f"{i}.png"))

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]




def run(dry_run):

    if not dry_run:
        # TODO multiprocessing
        import multiprocessing
        from multiprocessing.dummy import Pool as ThreadPool

        nthreads = 2

        filename = os.path.abspath(input_fn)
        pardir = os.path.abspath(__file__)
        # pardir = os.path.dirname(filename)

        tmpdir = tempfile.TemporaryDirectory(dir=pardir)
        os.makedirs(os.path.join(tmpdir.name, "pre"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir.name, "post"), exist_ok=True)
        imgs = pdfToImgs(filename, tmpdir.name)

        filteredImgs = batchFilter(tmpdir.name)

        imgsToPdf(filteredImgs, output_fn, tmpdir.name)

        tmpdir.cleanup()

    else:
        imgs = pdfToImgs(input_fn, None)

        p = Pipeline(imgs[0])

        import matplotlib.pyplot as plt
        fig, ((ax1, ax2, ax3, ax3a), (ax4, ax5, ax5a, ax6)) = plt.subplots(2, 4, figsize=(10, 5))

        plt.subplots_adjust(wspace=0, hspace=0.5)

        def draw_all():
            interpolation = "bilinear"
            ax1.imshow(p.im, cmap="gray", interpolation=interpolation)
            ax2.imshow(np.log(p.f_im + 1e-1), cmap="gray", interpolation=interpolation)
            ax3.imshow(p.g_im, interpolation=interpolation)
            ax3a.imshow(p.g_im_bg, interpolation=interpolation)
            ax4.imshow(p.notch, interpolation=interpolation)
            ax5.imshow(np.log(p.denotched_f + 1e-1), cmap="gray", interpolation=interpolation)
            ax5a.imshow(np.log(p.im_filtered + 1e-1), cmap="gray", interpolation=interpolation)
            ax6.imshow(p.result, cmap="gray", interpolation=interpolation)

            fig.canvas.draw_idle()

        def downscale_fn(downscale):
            p.rescale(downscale)
            p.fft()
            p.gaussFilter()
            p.findPeaks()
            p.backtransform()
            p.postProcess()
            draw_all()


        axcolor = 'lightgoldenrodyellow'

        divider = make_axes_locatable(ax1)
        ax_downscale = divider.append_axes("bottom", size="5%", pad=.05)
        scale_slider = Slider(
            ax=ax_downscale,
            label='Downsample %',
            valmin=0.1,
            valmax=1.0,
            valinit=p.downscale,
        )

        scale_slider.on_changed(downscale_fn)


        def downscale_peakfinder(scaleFactor):
            p.gaussFilter(scaleFactor)
            p.findPeaks()
            p.backtransform()
            p.postProcess()
            draw_all()

        divider = make_axes_locatable(ax2)
        ax_scaleFactor = divider.append_axes("bottom", size="5%", pad=.05)
        scaleFactor_slider = Slider(
            ax=ax_scaleFactor,
            label='ScaleFFT',
            valmin=1,
            valmax=10,
            valinit=p.scaleFactor,
        )

        scaleFactor_slider.on_changed(downscale_peakfinder)

        def on_maxpeaks(maxpeaks):
            p.findPeaks(maxpeaks)
            p.backtransform()
            p.postProcess()
            draw_all()

        divider = make_axes_locatable(ax4)
        ax_peaks = divider.append_axes("bottom", size="5%", pad=.05)
        maxpeaks_slider = Slider(
            ax=ax_peaks,
            label='max. Peaks',
            valmin=10,
            valmax=1000,
            valinit=p.maxpeaks,
        )

        maxpeaks_slider.on_changed(on_maxpeaks)

        def on_gamma(gamma):
            p.postProcess(gamma)
            draw_all()

        divider = make_axes_locatable(ax6)
        ax_gamma = divider.append_axes("bottom", size="5%", pad=.05)
        scaleFactor_slider = Slider(
            ax=ax_gamma,
            label='gamma',
            valmin=0.1,
            valmax=10,
            valinit=p.gamma,
        )

        scaleFactor_slider.on_changed(on_gamma)


        # run pipeline
        p.full_pipeline()
        draw_all()

        plt.show()

run(dry_run)