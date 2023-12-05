from functools import partial
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

# setup logger to show time
import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


from pikepdf import Pdf, PdfImage, Name
import zlib


def pdfToImgs(filename):
    logger.info("Opening PDF...")
    pdf = Pdf.open(filename)
    page = pdf.pages[0]
    # image_name, rawimage = next(page.images.items())

    imgs = []

    n_pages = len(pdf.pages)

    logger.info("Extracting pages from PDF...")
    for i, page in enumerate(tqdm(pdf.pages)):
        image_name, rawimage = next(page.images.items())

        pdfimage = PdfImage(rawimage)

        rawimage = pdfimage.obj
        pilImage = pdfimage.as_pil_image().convert("L")
        img = np.array(pilImage)
        img = cv2.resize(
            img,
            dsize=(int(img.shape[1] * downscale), int(img.shape[0] * downscale)),
            interpolation=cv2.INTER_CUBIC,
        )
        imgs.append(img)

    return imgs, n_pages


def save(file, tmpdir):
    imgs_fns = []
    for root, dirs, files in os.walk(os.path.join(tmpdir, "post")):
        for file_ in files:
            imgs_fns.append(os.path.join(root, file_))

    assert imgs_fns, "No images found"

    # multiple inputs (variant 2)
    if file.suffix == ".pdf":
        pdf = Pdf.new()
        for img_fn in tqdm(imgs_fns):
            img = Image.open(img_fn)
            img = np.array(img)
            img = cv2.resize(
                img,
                dsize=(int(img.shape[1] / downscale), int(img.shape[0] / downscale)),
                interpolation=cv2.INTER_CUBIC,
            )
            img = Image.fromarray(img)
            pdf.add_page().add_image(img)
        pdf.save(file)
    else:
        # save as images
        for img_fn in tqdm(imgs_fns, desc="Saving images"):
            img = Image.open(img_fn)
            img = np.array(img)
            img = cv2.resize(
                img,
                dsize=(int(img.shape[1] / downscale), int(img.shape[0] / downscale)),
                interpolation=cv2.INTER_CUBIC,
            )
            img = Image.fromarray(img)
            img.save(file)

    logger.info(f"Saved to {file}")


def run_pipeline(tmpdir):
    _, _, imgs = next(os.walk(os.path.join(tmpdir, "pre")))

    logger.info(f"Read N={len(imgs)} images from {os.path.join(tmpdir, 'pre')}")

    def get_pipeline(img):
        img = np.array(img)
        p = Pipeline(img)
        p.rescale()
        p.fft()
        p.gaussFilter()
        p.findPeaks()
        p.backtransform()
        p.post_process()

        return p

    for i, img_fn in tqdm(enumerate(imgs)):
        img = Image.open(os.path.join(tmpdir, "pre", img_fn))
        if img.mode == "RGB":
            channels = ["R", "G", "B"]
            pipelines = [get_pipeline(img.getchannel(c)) for c in channels]
            pipeline = pipelines[0]
            results = [p.result for p in pipelines]
            result = np.stack(results, axis=2)
        else:
            p = get_pipeline(img)
            result = p.result

        result = np.clip(np.array(result), 0, 255).astype(np.uint8)
        out_im = Image.fromarray(result)
        out_im.save(os.path.join(tmpdir, "post", f"{i}.png"))

    logger.info(f"Saved N={len(imgs)} images to {os.path.join(tmpdir, 'post')}")
    return pipeline


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [
        alist[i * length // wanted_parts : (i + 1) * length // wanted_parts]
        for i in range(wanted_parts)
    ]


def read_image(filename):
    pilImage = Image.open(filename)
    img = np.array(pilImage)
    img = cv2.resize(
        img,
        dsize=(int(img.shape[1] * downscale), int(img.shape[0] * downscale)),
        interpolation=cv2.INTER_CUBIC,
    )
    return [img], 1  # Wrap img in a list to mimic the output of pdfToImgs


### main code
from unchecker.pipeline import (
    downscale,
    scaleFactor,
    maxpeaks,
    gaussMain,
    gaussBG,
    maxpeaks,
    gamma,
)

dry_run = False

input_fn_default = "example.pdf"
input_fn_default = "Baumeister.jpg"

import fire

output_dir = Path(__file__).parents[1] / "out"
output_dir.mkdir(exist_ok=True)

input_dir = Path(__file__).parents[1]
input_fn = input_dir / input_fn_default
output_fn = output_dir / f"result{input_fn.suffix.lower()}"


def main(
    input_fn=input_fn,
    output_fn=output_fn,
    downscale=1.0,
    res_peakfinder=None,
    maxpeaks=1000,
    gauss_main=5,
    gauss_bg=10,
    gamma=1.0,
    dry_run=dry_run,
):

    pardir = Path(__file__).parent
    tmpdir_obj = tempfile.TemporaryDirectory(dir=pardir)

    # read the file
    filename = Path(input_fn).absolute()
    if filename.suffix.lower() in [".jpeg", ".jpg", ".png"]:
        imgs, n_pages = read_image(filename)
    else:
        imgs, n_pages = pdfToImgs(filename)

    os.makedirs(os.path.join(tmpdir_obj.name, "pre"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir_obj.name, "post"), exist_ok=True)

    if tmpdir_obj:
        for i, img in enumerate(imgs):
            Image.fromarray(img).save(
                os.path.join(tmpdir_obj.name, "pre", f"{i}.png")
            )

    logger.info(
        f"Saved N={len(imgs)} images to {os.path.join(tmpdir_obj.name, 'pre')}"
    )

    p = run_pipeline(tmpdir_obj.name)


    if not dry_run:
        save(output_fn, tmpdir_obj.name)
        tmpdir_obj.cleanup()

    else:
        import matplotlib.pyplot as plt


        mosaic = """
        aabchh
        aadehh
        aafghh
        """
        fig, axd = plt.subplot_mosaic(mosaic, layout="compressed")
        ax1 = axd["a"]
        ax2 = axd["b"]
        ax3 = axd["c"]
        ax3a = axd["d"]
        ax4 = axd["e"]
        ax5 = axd["f"]
        ax5a = axd["g"]
        ax6 = axd["h"]

        def draw_all():
            interpolation = "bilinear"
            ax1.imshow(p.im, cmap="gray", interpolation=interpolation)
            im2 = ax2.imshow(
                np.log(p.f_im + 1e-1), cmap="gray", interpolation=interpolation
            )
            im3 = ax3.imshow(p.g_im, interpolation=interpolation)
            vmin, vmax = im3.get_clim()

            ax3a.imshow(p.g_im_bg, interpolation=interpolation)
            ax4.imshow(1 - p.notch, interpolation=interpolation, cmap="Reds")
            im5 = ax5.imshow(
                np.log(p.denotched_f + 1e-1),
                interpolation=interpolation,
                vmin=vmin,
                vmax=vmax,
            )
            ax5a.imshow(
                np.log(p.im_filtered + 1e-1), cmap="gray", interpolation=interpolation
            )
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
                ax.tick_params(top=False, bottom=False)
                ax.set_xticklabels([])

            fig.canvas.draw_idle()

        def refreshETA(s, e):
            time_page = e - s
            time_document = time_page * n_pages
            fig.suptitle(
                f"ETA: {time_page:.2f}s per page, \n {int(time_document // 60)} min:{int(time_document % 60)}s for entire document"
            )

        def downscale_fn(downscale):
            start = time.process_time()
            p.rescale(downscale)
            p.fft()
            p.gaussFilter()
            p.findPeaks()
            p.backtransform()
            p.post_process()
            end = time.process_time()

            refreshETA(start, end)
            draw_all()

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, "Save settings", hovercolor="0.975")

        def saveSettings(event):
            p.save_settings()

        button.on_clicked(saveSettings)

        divider = make_axes_locatable(ax1)
        ax_downscale = divider.append_axes("bottom", size="5%", pad=0.05)
        scale_slider = Slider(
            ax=ax_downscale,
            label="Downsample %",
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
            p.post_process()
            end = time.process_time()
            refreshETA(start, end)
            draw_all()

        divider = make_axes_locatable(ax2)
        ax_scaleFactor = divider.append_axes("bottom", size="5%", pad=0.05)
        scaleFactor_slider = Slider(
            ax=ax_scaleFactor,
            label="ScaleFFT",
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
            p.post_process()
            end = time.process_time()
            refreshETA(start, end)
            draw_all()

        divider = make_axes_locatable(ax3)
        ax_scaleFactor = divider.append_axes("bottom", size="5%", pad=0.05)
        gaussMain_slider = Slider(
            ax=ax_scaleFactor,
            label="GaussMain",
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
            p.post_process()
            end = time.process_time()
            refreshETA(start, end)
            draw_all()

        divider = make_axes_locatable(ax3a)
        ax_scaleFactor = divider.append_axes("bottom", size="5%", pad=0.05)
        gaussBG_slider = Slider(
            ax=ax_scaleFactor,
            label="GaussBG",
            valmin=1,
            valmax=30,
            valinit=p.gaussBG,
        )

        gaussBG_slider.on_changed(set_gauss_bg)

        def on_maxpeaks(maxpeaks):
            start = time.process_time()
            p.findPeaks(maxpeaks)
            p.backtransform()
            p.post_process()
            end = time.process_time()
            refreshETA(start, end)
            draw_all()

        divider = make_axes_locatable(ax4)
        ax_peaks = divider.append_axes("bottom", size="5%", pad=0.05)
        maxpeaks_slider = Slider(
            ax=ax_peaks,
            label="max. Peaks",
            valmin=10,
            valmax=1000,
            valinit=p.maxpeaks,
        )

        maxpeaks_slider.on_changed(on_maxpeaks)

        def on_notch(notch_size):
            start = time.process_time()
            p.findPeaks(notch_size=notch_size)
            p.backtransform()
            p.post_process()
            end = time.process_time()
            refreshETA(start, end)
            draw_all()

        ax_notch = divider.append_axes("bottom", size="5%", pad=0.1)
        notchsize_slider = Slider(
            ax=ax_notch,
            label="notch_size",
            valmin=.1,
            valmax=10,
            valinit=p.notch_size,
        )

        notchsize_slider.on_changed(on_notch)

        def on_gamma(gamma):
            p.post_process(gamma)
            draw_all()

        divider = make_axes_locatable(ax6)
        ax_gamma = divider.append_axes("bottom", size="5%", pad=0.05)
        gamma_slider = Slider(
            ax=ax_gamma,
            label="gamma",
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

        fig.set_layout_engine(None)
        plt.show()


if __name__ == "__main__":
    # fire.Fire(main)

    in_dir = Path("/Users/jan/Library/CloudStorage/OneDrive-Personal/projects/citadels/characters")
    out_dir = Path("/Users/jan/Library/CloudStorage/OneDrive-Personal/projects/citadels/characters_decheckered")
    out_dir.mkdir(exist_ok=True)

    i =  0
    for file in in_dir.iterdir():
        # if i > 0: break
        if file.suffix.lower() in [".jpeg", ".jpg", ".png"]:
            if not "Prediger" in file.name: continue
            main(input_fn=file, output_fn=out_dir / file.name, dry_run=True)
            i += 1
