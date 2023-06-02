from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import multivariate_normal
import photutils
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import scipy as sp
import time
from tqdm import tqdm
import tempfile
import os
from PIL import Image

from pikepdf import Pdf, PdfImage, Name

import argparse

from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1 import make_axes_locatable

from unchecker.pipeline import *

import matplotlib

dry_run = True
use_settings = False
# input_fn_default = "example.pdf"
input_fn_default = "Baumeister.jpg"


from pikepdf import Pdf, PdfImage, Name
import zlib

def pdfToImgs(filename, tmpdir=None):
    print("Opening PDF...")
    pdf = Pdf.open(filename)
    page = pdf.pages[0]
    # image_name, rawimage = next(page.images.items())

    imgs = []

    n_pages = len(pdf.pages)

    print("Extracting pages from PDF...")
    for i, page in enumerate(tqdm(pdf.pages)):
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

    return imgs, n_pages


def imgsToPdf(imgs, filename, tmpdir):
    imgs_fns = []
    for root, dirs, files in os.walk(os.path.join(tmpdir, "post")):
        for file in files:
            imgs_fns.append(os.path.join(root, file))

    # multiple inputs (variant 2)
    with open(filename, "wb") as f:
        f.write(img2pdf.convert(imgs_fns))

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

def readImage(filename):
    pilImage = Image.open(filename).convert("L")
    img = np.array(pilImage)
    img = cv2.resize(img, dsize=(int(img.shape[1] * downscale), int(img.shape[0] * downscale)),
                     interpolation=cv2.INTER_CUBIC)
    return [img], 1  # Wrap img in a list to mimic the output of pdfToImgs


def run(dry_run):

    pardir = Path(__file__).parent
    tmpdir = tempfile.TemporaryDirectory(dir=pardir)


    filename = Path(input_fn_default).absolute()
    if filename.suffix.lower() in ['.jpeg', '.jpg', '.png']:
        imgs, n_pages = readImage(filename)
    else:
        imgs, n_pages = pdfToImgs(filename, tmpdir.name)


    if not dry_run:
        # TODO multiprocessing
        import multiprocessing
        from multiprocessing.dummy import Pool as ThreadPool

        nthreads = 2

        # pardir = os.path.dirname(filename)

        os.makedirs(os.path.join(tmpdir.name, "pre"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir.name, "post"), exist_ok=True)


        filteredImgs = batchFilter(tmpdir.name)

        print(output_fn)
        imgsToPdf(filteredImgs, output_fn, tmpdir.name)

        tmpdir.cleanup()

    else:
        p = Pipeline(imgs[0])

        import matplotlib.pyplot as plt
        fig, ((ax1, ax2, ax3, ax3a), (ax4, ax5, ax5a, ax6)) = plt.subplots(2, 4, figsize=(10, 5))

        plt.subplots_adjust(wspace=0, hspace=0.8)

        def draw_all():
            interpolation = "bilinear"
            ax1.imshow(p.im, cmap="gray", interpolation=interpolation)
            im2 = ax2.imshow(np.log(p.f_im + 1e-1), cmap="gray", interpolation=interpolation)
            im3 = ax3.imshow(p.g_im, interpolation=interpolation)
            vmin, vmax = im3.get_clim()

            ax3a.imshow(p.g_im_bg, interpolation=interpolation)
            ax4.imshow(1-p.notch, interpolation=interpolation, cmap="Reds")
            im5 = ax5.imshow(np.log(p.denotched_f + 1e-1), interpolation=interpolation, vmin=vmin, vmax=vmax)
            ax5a.imshow(np.log(p.im_filtered + 1e-1), cmap="gray", interpolation=interpolation)
            ax6.imshow(p.result, cmap="gray", interpolation=interpolation)

            ax1.set_title("Input")
            ax2.set_title("FFT spectrum")
            ax3.set_title("Denoised spectrum")
            ax3a.set_title("Background spectrum")

            ax4.set_title("Detected peaks")
            ax5.set_title("Cleaned spectrum")
            ax5a.set_title("Backtransformed image")
            ax6.set_title("Postprocessed image")

            for ax in plt.gcf().get_axes():
                ax.tick_params(top = False, bottom=False)
                ax.set_xticklabels([])

            fig.canvas.draw_idle()

        def refreshETA(s, e):
            time_page = e - s
            time_document = time_page * n_pages
            fig.suptitle(f"ETA: {time_page:.2f}s per page, \n {int(time_document // 60)} min:{int(time_document % 60)}s for entire document")

        def downscale_fn(downscale):
            start = time.process_time()
            p.rescale(downscale)
            p.fft()
            p.gaussFilter()
            p.findPeaks()
            p.backtransform()
            p.postProcess()
            end = time.process_time()

            refreshETA(start, end)
            draw_all()

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Save settings', hovercolor='0.975')

        def saveSettings(event):
            p.saveSettings()

        button.on_clicked(saveSettings)

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
            start = time.process_time()
            p.gaussFilter(scaleFactor)
            p.findPeaks()
            p.backtransform()
            p.postProcess()
            end = time.process_time()
            refreshETA(start, end)
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

        def set_gauss_main(gaussMain):
            start = time.process_time()
            p.gaussFilter(gaussMain=gaussMain)
            p.findPeaks()
            p.backtransform()
            p.postProcess()
            end = time.process_time()
            refreshETA(start, end)
            draw_all()


        divider = make_axes_locatable(ax3)
        ax_scaleFactor = divider.append_axes("bottom", size="5%", pad=.05)
        gaussMain_slider = Slider(
            ax=ax_scaleFactor,
            label='GaussMain',
            valmin=1,
            valmax=10,
            valinit=p.gaussMain,
        )

        gaussMain_slider.on_changed(set_gauss_main)

        def set_gauss_bg(gaussBG):
            start = time.process_time()
            p.gaussFilter(gaussBG=gaussBG)
            p.findPeaks()
            p.backtransform()
            p.postProcess()
            end = time.process_time()
            refreshETA(start, end)
            draw_all()


        divider = make_axes_locatable(ax3a)
        ax_scaleFactor = divider.append_axes("bottom", size="5%", pad=.05)
        gaussBG_slider = Slider(
            ax=ax_scaleFactor,
            label='GaussBG',
            valmin=1,
            valmax=30,
            valinit=p.gaussBG,
        )

        gaussBG_slider.on_changed(set_gauss_bg)

        def on_maxpeaks(maxpeaks):
            start = time.process_time()
            p.findPeaks(maxpeaks)
            p.backtransform()
            p.postProcess()
            end = time.process_time()
            refreshETA(start, end)
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
        gamma_slider = Slider(
            ax=ax_gamma,
            label='gamma',
            valmin=0.1,
            valmax=5,
            valinit=p.gamma,
        )

        gamma_slider.on_changed(on_gamma)


        # run pipeline
        s = time.process_time()
        p.full_pipeline()
        e = time.process_time()
        refreshETA(s, e)
        draw_all()
        # plt.tight_layout()

        plt.show()


### main code
def main():
    from unchecker.pipeline import downscale, scaleFactor, maxpeaks, gaussMain, gaussBG, maxpeaks, gamma

    parser = argparse.ArgumentParser(description='An fft-based grid removal filter')
    parser.add_argument('--input', '-i', default="", type=str,
                    help='Input file path. Accepts .pdf, .jpeg, .jpg, .png files')


    parser.add_argument('--output', '-o', default="", type=str,
                        help='Output file path. ')

    parser.add_argument('--downscale', '-d', default=downscale, type=float,
                        help='Downsamples the PDF, given in percent. Choose as large as possible in terms of computational cost. ')
    parser.add_argument('--res_peakfinder', '-r', default=scaleFactor**-1, type=float,
                        help="Downsamples FFT spectrum on which the peakfinder acts, accelerating it. Doesn't affect output resolution. Choose as low as possible for optimal speed. ")
    parser.add_argument('--maxpeaks', '-m', default=maxpeaks, type=int,
                        help='Maximum number of peaks for peakfinder. Larger values give better grid removals at the cost of overall image degradation. ')
    parser.add_argument('--gauss_main', '-gm', default=gaussMain, type=int,
                        help='Kernel size to apply to the noisy FFT spectrum.  ')
    parser.add_argument('--gauss_bg', '-gbg', default=maxpeaks, type=int,
                        help='Kernel size to apply to estimate the background for the peakfinding algorithm.  ')

    parser.add_argument('--gamma', '-g', default=gamma, type=float,
                        help='Increases BW contrast in image in the end. Choose larger for higher contrast. ')

    parser.add_argument('--no_dry_run', '-ndr', default=False, type=bool,
                        help='Ratio for middle preservation. ')

    parser.add_argument('--use_settings', '-u', default=False, type=bool,
                        help='Use settings previously determined from dry-run. ')

    args = parser.parse_args()

    input_fn = args.input if args.input else input_fn_default
    output_fn = args.output
    downscale = args.downscale
    scaleFactor = int(downscale**(-1))
    gamma = args.gamma
    maxpeaks = args.maxpeaks
    # debug = args.dry_
    dry_run = not args.no_dry_run
    use_settings = args.use_settings
    print(input_fn)
    run(dry_run)

if __name__ == '__main__':
    main()