'''
   
    Tiling code inspired from GTP:
   https://github.com/vkola-lab/tmi2022/blob/main

 
'''

from __future__ import print_function
import json
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from optparse import OptionParser
import re
import shutil
from unicodedata import normalize
import numpy as np
import subprocess
from glob import glob
from multiprocessing import Process, JoinableQueue
import time
import os
import sys

from xml.dom import minidom
from PIL import Image, ImageDraw, ImageCms
from skimage import color, io
Image.MAX_IMAGE_PIXELS = None
# from imageio import imwrite as imsave
# from imageio import imread

VIEWER_SLIDE_NAME = 'slide'


class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds, quality, _Bkg, _ROIpc):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._slide = None
        self._Bkg = _Bkg
        self._ROIpc = _ROIpc

    def RGB_to_lab(self, tile):
        print("RGB to Lab")
        Lab = color.rgb2lab(tile)
        return Lab

    def Lab_to_RGB(self, Lab):
        print("Lab to RGB")
        newtile = (color.lab2rgb(Lab) * 255).astype(np.uint8)
        return newtile

    def normalize_tile(self, tile, NormVec):
        Lab = self.RGB_to_lab(tile)
        TileMean = [0, 0, 0]
        TileStd = [1, 1, 1]
        newMean = NormVec[0:3] 
        newStd = NormVec[3:6]
        for i in range(3):
            TileMean[i] = np.mean(Lab[:, :, i])
            TileStd[i] = np.std(Lab[:, :, i])
            # print("mean/std chanel " + str(i) + ": " + str(TileMean[i]) + " / " + str(TileStd[i]))
            tmp = ((Lab[:, :, i] - TileMean[i]) * (newStd[i] / TileStd[i])) + newMean[i]
            if i == 0:
                tmp[tmp < 0] = 0 
                tmp[tmp > 100] = 100 
                Lab[:, :, i] = tmp
            else:
                tmp[tmp < -128] = 128 
                tmp[tmp > 127] = 127 
                Lab[:, :, i] = tmp
        tile = self.Lab_to_RGB(Lab)
        return tile

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            #associated, level, address, outfile = data
            associated, level, address, outfile, format, outfile_bw, PercentMasked, SaveMasks, TileMask, Normalize = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            #try:
            if True:
                try:
                    tile = dz.get_tile(level, address)
                    # A single tile is being read
                    #check the percentage of the image with "information". Should be above 50%
                    gray = tile.convert('L')
                    bw = gray.point(lambda x: 0 if x < 220 else 1, 'F')
                    arr = np.array(np.asarray(bw))
                    avgBkg = np.average(bw)
                    bw = gray.point(lambda x: 0 if x < 220 else 1, '1')
                    # check if the image is mostly background
                    print("res: " + outfile + " is " + str(avgBkg))
                    if avgBkg <= (self._Bkg / 100.0):
                        # print("PercentMasked: %.6f, %.6f" % (PercentMasked, self._ROIpc / 100.0) )
                        # if an Aperio selection was made, check if is within the selected region
                        if PercentMasked >= (self._ROIpc / 100.0):

                            if Normalize != '':
                                print("normalize " + str(outfile))
                                # arrtile = np.array(tile)
                                tile = Image.fromarray(self.normalize_tile(tile, Normalize).astype('uint8'), 'RGB')

                            tile.save(outfile, quality=self._quality)
                            if bool(SaveMasks) == True:
                                height = TileMask.shape[0]
                                width = TileMask.shape[1]
                                TileMaskO = np.zeros((height, width, 3), 'uint8')
                                maxVal = float(TileMask.max())
                                TileMaskO[..., 0] = (TileMask[:, :].astype(float) / maxVal * 255.0).astype(int)
                                TileMaskO[..., 1] = (TileMask[:, :].astype(float) / maxVal * 255.0).astype(int)
                                TileMaskO[..., 2] = (TileMask[:, :].astype(float) / maxVal * 255.0).astype(int)
                                TileMaskO = np.array(Image.fromarray(TileMaskO).resize((arr.shape[0], arr.shape[1])))
                                # TileMaskO = imresize(TileMaskO, (arr.shape[0], arr.shape[1],3))
                                TileMaskO[TileMaskO < 10] = 0
                                TileMaskO[TileMaskO >= 10] = 255
                                # Use PIL to save the mask instead of imsave
                                Image.fromarray(TileMaskO).save(outfile_bw, quality=self._quality)

                    self._queue.task_done()
                except Exception as e:
                    # print(level, address)
                    print("image %s failed at dz.get_tile for level %f" % (self._slidepath, level))
                    # e = sys.exc_info()[0]
                    print(e)
                    self._queue.task_done()

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, format, associated, queue, slide, basenameJPG, xmlfile, mask_type, xmlLabel, ROIpc, ImgExtension, SaveMasks, Mag, normalize, Fieldxml):
        self._dz = dz
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._slide = slide
        self._xmlfile = xmlfile
        self._mask_type = mask_type
        self._xmlLabel = xmlLabel
        self._ROIpc = ROIpc
        self._ImgExtension = ImgExtension
        self._SaveMasks = SaveMasks
        self._Mag = Mag
        self._normalize = normalize
        self._Fieldxml = Fieldxml

    def run(self):
        self._write_tiles()
        self._write_dzi()

    def _write_tiles(self):
        Magnification = 20
        tol = 2
        #get slide dimensions, zoom levels, and objective information
        Factors = self._slide.level_downsamples
        try:
            Objective = float(self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            # print(self._basename + " - Obj information found")
        except:
            print(self._basename + " - No Obj information found")
            return

        #calculate magnifications
        Available = tuple(Objective / x for x in Factors)
        #find highest magnification greater than or equal to 'Desired'
        Mismatch = tuple(x-Magnification for x in Available)
        AbsMismatch = tuple(abs(x) for x in Mismatch)
        if len(AbsMismatch) < 1:
            print(self._basename + " - Objective field empty!")
            return

        xml_valid = False
        # a dir was provided for xml files

        if True:
            ImgID = os.path.basename(self._basename)
            xmldir = os.path.join(self._xmlfile, ImgID + '.xml')
            # print("xml:")
            # print(xmldir)
            if (self._xmlfile != ''):
                # print("read xml file...")
                mask, xml_valid, Img_Fact = self.xml_read(xmldir, self._xmlLabel, self._Fieldxml)
                if xml_valid == False:
                    print("Error: xml %s file cannot be read properly - please check format" % xmldir)
                    return

            # print("current directory: %s" % self._basename)

            #return
            #print(self._dz.level_count)

            for level in range(self._dz.level_count-1, -1, -1):
                ThisMag = Available[0]/pow(2, self._dz.level_count-(level+1))
                if self._Mag > 0:
                    if ThisMag != self._Mag:
                        continue
                ########################################
                #tiledir = os.path.join("%s_files" % self._basename, str(level))
                tiledir = os.path.join("%s_files" % self._basename, str(ThisMag))
                if not os.path.exists(tiledir):
                    os.makedirs(tiledir)
                cols, rows = self._dz.level_tiles[level]
                
                for row in range(rows):
                    for col in range(cols):
                        InsertBaseName = False
                        if InsertBaseName:
                          tilename = os.path.join(tiledir, '%s_%d_%d.%s' % (
                                          self._basenameJPG, col, row, self._format))
                          tilename_bw = os.path.join(tiledir, '%s_%d_%d_mask.%s' % (
                                          self._basenameJPG, col, row, self._format))
                        else:
                          tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                          col, row, self._format))
                          tilename_bw = os.path.join(tiledir, '%d_%d_mask.%s' % (
                                          col, row, self._format))
                        if xml_valid:
                            Dlocation, Dlevel, Dsize = self._dz.get_tile_coordinates(level, (col, row))
                            Ddimension = tuple([pow(2, (self._dz.level_count - 1 - level)) * x for x in self._dz.get_tile_dimensions(level, (col, row))])
                            startIndY_current_level_conv = (int((Dlocation[1]) / Img_Fact))
                            endIndY_current_level_conv = (int((Dlocation[1] + Ddimension[1]) / Img_Fact))
                            startIndX_current_level_conv = (int((Dlocation[0]) / Img_Fact))
                            endIndX_current_level_conv = (int((Dlocation[0] + Ddimension[0]) / Img_Fact))
                            
                            TileMask = mask[startIndY_current_level_conv:endIndY_current_level_conv, startIndX_current_level_conv:endIndX_current_level_conv]
                            PercentMasked = mask[startIndY_current_level_conv:endIndY_current_level_conv, startIndX_current_level_conv:endIndX_current_level_conv].mean() 

                            if self._mask_type == 0:
                                # keep ROI outside of the mask
                                PercentMasked = 1.0 - PercentMasked
                        else:
                            PercentMasked = 1.0
                            TileMask = []

                        if not os.path.exists(tilename):
                            self._queue.put((self._associated, level, (col, row),
                                        tilename, self._format, tilename_bw, PercentMasked, self._SaveMasks, TileMask, self._normalize))
                        self._tile_done()

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

    def _write_dzi(self):
        with open('%s.dzi' % self._basename, 'w') as fh:
            fh.write(self.get_dzi())

    def get_dzi(self):
        return self._dz.get_dzi(self._format)

    def xml_read(self, xmldir, Attribute_Name, Fieldxml):
        # Original size of the image
        ImgMaxSizeX_orig = float(self._dz.level_dimensions[-1][0])
        ImgMaxSizeY_orig = float(self._dz.level_dimensions[-1][1])
        # Number of centers at the highest resolution
        cols, rows = self._dz.level_tiles[-1]

        NewFact = max(ImgMaxSizeX_orig, ImgMaxSizeY_orig) / min(max(ImgMaxSizeX_orig, ImgMaxSizeY_orig), 15000.0)
        Img_Fact = float(ImgMaxSizeX_orig) / 5.0 / float(cols)
       
        try:
            xmlcontent = minidom.parse(xmldir)
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid, 1.0

        xy = {}
        xy_neg = {}
        NbRg = 0
        labelIDs = xmlcontent.getElementsByTagName('Annotation')
        # print("%d labels" % len(labelIDs) )
        for labelID in labelIDs:
            if (Attribute_Name==[]) | (Attribute_Name==''):
                    isLabelOK = True
            else:
                try:
                    labeltag = labelID.getElementsByTagName('Attribute')[0]
                    if (Attribute_Name==labeltag.attributes[Fieldxml].value):
                        isLabelOK = True
                    else:
                        isLabelOK = False
                except:
                    isLabelOK = False
            if Attribute_Name == "non_selected_regions":
                isLabelOK = True

            if isLabelOK:
                regionlist = labelID.getElementsByTagName('Region')
                for region in regionlist:
                    vertices = region.getElementsByTagName('Vertex')
                    NbRg += 1
                    regionID = region.attributes['Id'].value + str(NbRg)
                    NegativeROA = region.attributes['NegativeROA'].value
                    # print("%d vertices" % len(vertices))
                    if len(vertices) > 0:
                        #print( len(vertices) )
                        if NegativeROA=="0":
                            xy[regionID] = []
                            for vertex in vertices:
                                x = int(round(float(vertex.attributes['X'].value) / NewFact))
                                y = int(round(float(vertex.attributes['Y'].value) / NewFact))
                                xy[regionID].append((x,y))
    
                        elif NegativeROA=="1":
                            xy_neg[regionID] = []
                            for vertex in vertices:
                                x = int(round(float(vertex.attributes['X'].value) / NewFact))
                                y = int(round(float(vertex.attributes['Y'].value) / NewFact))
                                xy_neg[regionID].append((x,y))

        img = Image.new('L', (int(ImgMaxSizeX_orig/NewFact), int(ImgMaxSizeY_orig/NewFact)), 0)
        for regionID in xy.keys():
            xy_a = xy[regionID]
            ImageDraw.Draw(img, 'L').polygon(xy_a, outline=255, fill=255)
        for regionID in xy_neg.keys():
            xy_a = xy_neg[regionID]
            ImageDraw.Draw(img, 'L').polygon(xy_a, outline=255, fill=0)
        
        mask = np.array(img)
        
        if Attribute_Name == "non_selected_regions":
            Image.fromarray(255-mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg"))
        else:
           if self._mask_type==0:
               Image.fromarray(255-mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + "_inv.jpeg"))
           else:
               Image.fromarray(mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg"))  
        
        return mask / 255.0, xml_valid, NewFact


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, format, tile_size, overlap,
                limit_bounds, quality, workers, with_viewer, Bkg, basenameJPG, xmlfile, mask_type, ROIpc, oLabel, ImgExtension, SaveMasks, Mag, normalize, Fieldxml):
        if with_viewer:
            # Check extra dependency before doing a bunch of work
            import jinja2
        
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._xmlfile = xmlfile
        self._mask_type = mask_type
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._with_viewer = with_viewer
        self._Bkg = Bkg
        self._ROIpc = ROIpc
        self._dzi_data = {}
        self._xmlLabel = oLabel
        self._ImgExtension = ImgExtension
        self._SaveMasks = SaveMasks
        self._Mag = Mag
        self._normalize = normalize
        self._Fieldxml = Fieldxml

        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                limit_bounds, quality, self._Bkg, self._ROIpc).start()

    def run(self):
        self._run_image()
        if self._with_viewer:
            for name in self._slide.associated_images:
                self._run_image(name)
            self._write_html()
            self._write_static()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            if self._with_viewer:
                 basename = os.path.join(self._basename, VIEWER_SLIDE_NAME)
            else:
                basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        # print("enter DeepZoomGenerator")
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)
        # print("enter DeepZoomImageTiler")
        tiler = DeepZoomImageTiler(dz, basename, self._format, associated, self._queue, self._slide, self._basenameJPG, self._xmlfile, self._mask_type, self._xmlLabel, self._ROIpc, self._ImgExtension, self._SaveMasks, self._Mag, self._normalize, self._Fieldxml)
        tiler.run()
        self._dzi_data[self._url_for(associated)] = tiler.get_dzi()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _write_html(self):
        import jinja2
        env = jinja2.Environment(loader=jinja2.PackageLoader(__name__), autoescape=True)
        template = env.get_template('slide-multipane.html')
        associated_urls = dict((n, self._url_for(n))
                for n in self._slide.associated_images)
        try:
            mpp_x = self._slide.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = self._slide.properties[openslide.PROPERTY_NAME_MPP_Y]
            mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            mpp = 0
        # Embed the dzi metadata in the HTML to work around Chrome's
        # refusal to allow XmlHttpRequest from file:///, even when
        # the originating page is also a file:///
        data = template.render(slide_url=self._url_for(None), slide_mpp=mpp, associated=associated_urls, properties=self._slide.properties, dzi_data=json.dumps(self._dzi_data))
        with open(os.path.join(self._basename, 'index.html'), 'w') as fh:
            fh.write(data)

    def _write_static(self):
        basesrc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'static')
        basedst = os.path.join(self._basename, 'static')
        self._copydir(basesrc, basedst)
        self._copydir(os.path.join(basesrc, 'images'),
                os.path.join(basedst, 'images'))

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()


def xml_read_labels(xmldir, Fieldxml):
        try:
            xmlcontent = minidom.parse(xmldir)
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid
        labeltag = xmlcontent.getElementsByTagName('Attribute')
        xml_labels = []
        for xmllabel in labeltag:
            xml_labels.append(xmllabel.attributes[Fieldxml].value)
        if xml_labels==[]:
            xml_labels = ['']
        # print(xml_labels)
        return xml_labels, xml_valid 


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <slide>')

    parser.add_option('-L', '--ignore-bounds', dest='limit_bounds',
        default=True, action='store_false',
        help='display entire scan area')
    parser.add_option('-e', '--overlap', metavar='PIXELS', dest='overlap',
        type='int', default=1,
        help='overlap of adjacent tiles [1]')
    parser.add_option('-f', '--format', metavar='{jpeg|png}', dest='format',
        default='jpeg',
        help='image format for tiles [jpeg]')
    parser.add_option('-j', '--jobs', metavar='COUNT', dest='workers',
        type='int', default=4,
        help='number of worker processes to start [4]')
    parser.add_option('-o', '--output', metavar='NAME', dest='basename',
        help='base name of output file')
    parser.add_option('-Q', '--quality', metavar='QUALITY', dest='quality',
        type='int', default=90,
        help='JPEG compression quality [90]')
    parser.add_option('-r', '--viewer', dest='with_viewer',
        action='store_true',
        help='generate directory tree with HTML viewer')
    parser.add_option('-s', '--size', metavar='PIXELS', dest='tile_size',
        type='int', default=254,
        help='tile size [254]')
    parser.add_option('-B', '--Background', metavar='PIXELS', dest='Bkg',
        type='float', default=50,
        help='Max background threshold [50]; percentager of background allowed')
    parser.add_option('-x', '--xmlfile', metavar='NAME', dest='xmlfile',
        help='xml file if needed')
    parser.add_option('-F', '--Fieldxml', metavar='{Name|Value}', dest='Fieldxml',
        default='Value',
        help='which field of the xml file is the label saved')
    parser.add_option('-m', '--mask_type', metavar='COUNT', dest='mask_type',
        type='int', default=1,
        help='if xml file is used, keep tile within the ROI (1) or outside of it (0)')
    parser.add_option('-R', '--ROIpc', metavar='PIXELS', dest='ROIpc',
        type='float', default=50,
        help='To be used with xml file - minimum percentage of tile covered by ROI (white)')
    parser.add_option('-l', '--oLabelref', metavar='NAME', dest='oLabelref',
        help='To be used with xml file - Only tile for label which contains the characters in oLabel')
    parser.add_option('-S', '--SaveMasks', metavar='NAME', dest='SaveMasks',
        default=False,
        help='set to yes if you want to save ALL masks for ALL tiles (will be saved in same directory with <mask> suffix)')
    parser.add_option('-M', '--Mag', metavar='PIXELS', dest='Mag',
        type='float', default=-1,
        help='Magnification at which tiling should be done (-1 of all)')
    parser.add_option('-N', '--normalize', metavar='NAME', dest='normalize',
        help='if normalization is needed, N list the mean and std for each channel. For example \'57,22,-8,20,10,5\' with the first 3 numbers being the targeted means, and then the targeted stds')

    (opts, args) = parser.parse_args()

    try:
        slidepath = args[0]
    except IndexError:
        parser.error('Missing slide argument')
    if opts.basename is None:
        opts.basename = os.path.splitext(os.path.basename(slidepath))[0]
    if opts.xmlfile is None:
        opts.xmlfile = ''

    try:
        if opts.normalize is not None:
            opts.normalize = [float(x) for x in opts.normalize.split(',')]
            if len(opts.normalize) != 6:
                opts.normalize = ''
                parser.error("ERROR: NO NORMALIZATION APPLIED: input vector does not have the right length - 6 values expected")
        else:
            opts.normalize = ''
    except:
        opts.normalize = ''
        parser.error("ERROR: NO NORMALIZATION APPLIED: input vector does not have the right format")

    # Get images from the slidepath
    files = glob(slidepath)
    ImgExtension = slidepath.split('*')[-1]
    
    # Check if we're dealing with SVS files only
    if '.svs' not in ImgExtension.lower():
        print("Error: This script only processes SVS files. Please provide a path to SVS files.")
        sys.exit(1)
    
    files = sorted(files)
    for imgNb in range(len(files)):
        filename = files[imgNb]
        opts.basenameJPG = os.path.splitext(os.path.basename(filename))[0]
        print("Processing SVS file: " + opts.basenameJPG)
        
        # Process with XML if provided
        if opts.xmlfile != '':
            xmldir = os.path.join(opts.xmlfile, opts.basenameJPG + '.xml')
            if os.path.isfile(xmldir):
                if (opts.mask_type == 1) or (opts.oLabelref != ''):
                    # either mask inside ROI, or mask outside but a reference label exist
                    xml_labels, xml_valid = xml_read_labels(xmldir, opts.Fieldxml)
                    if (opts.mask_type == 1):
                        # No inverse mask
                        Nbr_ROIs_ForNegLabel = 1
                    elif (opts.oLabelref != ''):
                        # Inverse mask and a label reference exist
                        Nbr_ROIs_ForNegLabel = 0

                    for oLabel in xml_labels:
                        if (opts.oLabelref in oLabel) or (opts.oLabelref == ''):
                            # is a label is identified 
                            if (opts.mask_type == 0):
                                # Inverse mask and label exist in the image
                                Nbr_ROIs_ForNegLabel += 1
                                # there is a label, and map is to be inverted
                                output = os.path.join(opts.basename, oLabel + '_inv', opts.basenameJPG)
                                if not os.path.exists(os.path.join(opts.basename, oLabel + '_inv')):
                                    os.makedirs(os.path.join(opts.basename, oLabel + '_inv'))
                            else:
                                Nbr_ROIs_ForNegLabel += 1
                                output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
                                if not os.path.exists(os.path.join(opts.basename, oLabel)):
                                    os.makedirs(os.path.join(opts.basename, oLabel))
                            
                            try:
                                DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, 
                                                   opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, 
                                                   opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, 
                                                   opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag, 
                                                   opts.normalize, opts.Fieldxml).run()
                            except Exception as e:
                                print(f"Failed to process file {filename}, error: {str(e)}")
                                
                        if Nbr_ROIs_ForNegLabel == 0:
                            print(f"label {opts.oLabelref} is not in that image; invert everything")
                            # a label ref was given, and inverse mask is required but no ROI with this label in that map --> take everything
                            oLabel = opts.oLabelref
                            output = os.path.join(opts.basename, opts.oLabelref + '_inv', opts.basenameJPG)
                            if not os.path.exists(os.path.join(opts.basename, oLabel + '_inv')):
                                os.makedirs(os.path.join(opts.basename, oLabel + '_inv'))
                            
                            try:
                                DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, 
                                                   opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, 
                                                   opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, 
                                                   opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag, 
                                                   opts.normalize, opts.Fieldxml).run()
                            except Exception as e:
                                print(f"Failed to process file {filename}, error: {str(e)}")
                else:
                    # Background
                    oLabel = "non_selected_regions"
                    output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
                    if not os.path.exists(os.path.join(opts.basename, oLabel)):
                        os.makedirs(os.path.join(opts.basename, oLabel))
                    try:
                        DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, 
                                           opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, 
                                           opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, 
                                           opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag, 
                                           opts.normalize, opts.Fieldxml).run()
                    except Exception as e:
                        print(f"Failed to process file {filename}, error: {str(e)}")
            else:
                print(f"No xml file found for slide {opts.basenameJPG}.svs (expected: {xmldir}). Directory or xml file does not exist")
                # Still process the SVS file without XML
                output = os.path.join(opts.basename, opts.basenameJPG)
                if os.path.exists(output + "_files"):
                    print(f"Image {opts.basenameJPG} already tiled")
                    continue
                try:
                    DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, 
                                       opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, 
                                       opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, 
                                       opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag, 
                                       opts.normalize, opts.Fieldxml).run()
                except Exception as e:
                    print(f"Failed to process file {filename}, error: {str(e)}")
        else:
            # No XML file, just process the SVS file
            output = os.path.join(opts.basename, opts.basenameJPG)
            if os.path.exists(output + "_files"):
                print(f"Image {opts.basenameJPG} already tiled")
                continue
            try:
                DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, 
                                   opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, 
                                   opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, 
                                   opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag, 
                                   opts.normalize, opts.Fieldxml).run()
            except Exception as e:
                print(f"Failed to process file {filename}, error: {str(e)}")

    print("End")
