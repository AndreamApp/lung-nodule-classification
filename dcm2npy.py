'''
Created by Wang Qiu Li
7/3/2018

get dicom info according to malignancy.csv and ld_scan.txt
'''

import os
import cv2
import numpy as np
import sys
import tools
from PIL import Image

truncate_upper_bound = 400
if len(sys.argv) > 1:
    truncate_upper_bound = int(sys.argv[1])

#basedir = '/data0/LIDC/DOI/'
basedir = tools.data_path
npy_path = 'truncate_%d/NPY/' % truncate_upper_bound  # tools.npy_path
nodule_shape = (10, 40, 40)



# LUNA2016 data prepare ,first step: truncate HU to -1000 to 400
def truncate_hu(image_array):
    image_array[image_array > truncate_upper_bound] = 0
    image_array[image_array < -1000] = 0
    return image_array
    
# LUNA2016 data prepare ,second step: normalzation the HU
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def cutTheImage(x, y, width, height, pix):
    x1 = x - int(width/2)
    x2 = x + int(width/2)
    y1 = y - int(height/2)
    y2 = y + int(height/2)
    img_cut = pix[x1:x2, y1:y2]
    return img_cut

def save_npy(resdir, nodule, shape):
    slices = nodule.get_all_slices(basedir)
    x_loc, y_loc, z_loc = nodule.x_loc, nodule.y_loc, nodule.z_loc

    # add z loc
    zstart = z_loc - 1 - int(shape[0]/2)
    zend = z_loc - 1 + int(shape[0]/2)

    cut_img = []
    tempsign = 0
    for zslice in slices[zstart: zend]:
        pix = zslice.pixel_array
        pix.flags.writeable = True

        # cut first or normalization first make a difference
        pix = truncate_hu(pix)
        pix = normalazation(pix)
        pix = cutTheImage(y_loc, x_loc, shape[1], shape[2], pix)
        pix = cv2.resize(pix, (20, 20))

        # scipy.misc.imsave(str(tempsign) + '.jpeg', cutpix)
        tempsign += 1
        cut_img.append(pix)

    level = round(float(nodule.maligancy))

    if level == 1:
        np.save(resdir + nodule.id + '_low' + '.npy', cut_img)
    elif level == 2:
        np.save(resdir + nodule.id + '_low' + '.npy', cut_img)
    elif level == 4:
        np.save(resdir + nodule.id + '_high' + '.npy', cut_img)
    elif level == 5:
        np.save(resdir + nodule.id + '_high' + '.npy', cut_img)

def save_all_npy(resdir, shape):
    tools.mkdirs(resdir)
    nodule_list = tools.get_nodules()
    for nodule in nodule_list:
        try:
            save_npy(resdir, nodule, shape)
            print(nodule)
        except Exception as e:
            # image area out of index. eg. shape=(10,40,40), nodule position=(2,10,10)
            tools.err(nodule)
            tools.err(e)


def angle_transpose(file,degree,flag_string):
    '''
     @param file : a npy file which store all information of one cubic
     @param degree: how many degree will the image be transposed,90,180,270 are OK
     @flag_string:  which tag will be added to the filename after transposed
    '''
    array = np.load(file)
    array = array.transpose(2, 1, 0)  # from x,y,z to z,y,x

    newarr = np.zeros(array.shape,dtype=np.float32)
    for depth in range(array.shape[0]):
        jpg = array[depth]
        jpg.reshape((jpg.shape[0],jpg.shape[1],1))
        img = Image.fromarray(jpg)
        #img.show()
        out = img.rotate(degree)
        newarr[depth,:,:] = np.array(out).reshape(array.shape[1],-1)[:,:]
    newarr = newarr.transpose(2, 1, 0)
    print(newarr.shape)
    np.save(file.replace(".npy",flag_string+".npy"),newarr)

def transpose_all_npy(resdir):
    filelist = os.listdir(resdir)
    for onefile in filelist:
        try:
            if 'high' in onefile:
                angle_transpose(resdir + onefile, 90, "_leftright")
                angle_transpose(resdir + onefile, 180, "_updown")
                angle_transpose(resdir + onefile, 270, "_diagonal")
            elif 'low' in onefile:
                angle_transpose(resdir + onefile, 270, "_diagonal")
            print('transpose %s successful', onefile)
        except BaseException as e:
            tools.err(onefile)
            tools.err(e)
            os.remove(resdir + onefile)


if __name__ == '__main__':
    save_all_npy(npy_path, nodule_shape)
    transpose_all_npy(npy_path)

