import png
import random
import numpy

class Colorgen(object):
    def __init__(self):
        self.background = (0, 0, 0)
        self.band_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.min_color = 100
        self.randomizer = random.Random(17)

    def _add_color(self):
        color = [self.randomizer.uniform(self.min_color, 255) for _ in range(3)]
        self.band_colors.append(tuple(color))
        return tuple(color)

    def _get_color(self, band):
        if band < len(self.band_colors):
            return self.band_colors[band]
        self._add_color()
        return self._get_color(band)

    def colors(self, bands):
        return [_get_color(self, i) for i in range(bands)]


class BlockstateDrawer(object):
    def __init__(self, filepath, colorgen):
        self.filepath = filepath
        self.colorgen = colorgen
        self.file_num = 0
        self.pixel_size = 20

    def save(self, *blockstates):
        imgs = []
        for blockstate in blockstates:
            imgs.append(self.colorize(blockstate))
        pixels = self.combine(imgs)
        pixels = self.stretch(pixels)

    def produce(self, img):
        pngimg = png.from_array(img.tolist(), mode='RGB')
        fname = self.filepath+str(file_num)+'.png'
        pngimg.save(fname)

    def colorize(self, blockstate):
        h, w, channels = blockstate.shape
        colors = self.colorgen.colors(channels)

        shaped = blockstate.respahe((h, w, channels, 1))
        colormat = [[colors]]
        colored = numpy.sum(shaped*colormat, 2)
        maxcolors = numpy.max(colored, 2)
        colored[maxed>255] = colored[maxed>255] / numpy.expand_dims(maxed[maxed>255], 1)
        return colored.astype(int)

    def combine(self, imgs):
        padded = self.pad_images(imgs)
        return numpy.concatenate(imgs, 1)

    def pad_images(self, imgs):
        maxh = max(img.shape[0] for img in imgs)
        maxw = max(img.shape[1] for img in imgs)
        return [self.pad_img(img, maxh, maxw+1) for img in imgs]

    def pad_img(self, img, new_h, new_w):
        h, w, c = img.shape
        pads = [(0, new_h-h), (0, new_w-w), (0, 0)]
        return img.pad(img, pads, 'constant', constant_values=(255,))

    def stretch(self, img):
        img = numpy.repeat(img, self.pixel_size, axis=0)
        return numpy.repeat(img, self.pixel_size, axis=1)


