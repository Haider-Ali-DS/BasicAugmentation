import glob
import skimage
from skimage.io import imread, imshow, imsave
from skimage.transform import rotate
from skimage import exposure
import numpy as np
import time
import concurrent.futures
import cv2
import multiprocessing as mp

img_dir = r"C:\Users\Haider Ali\Desktop\final_train_augment"
images = glob.glob(img_dir + "\\*\\*g")

def augment_pipeline(img_arr, angle, im_name, im_path):
    img_arr = rotate_and_save(img_arr, angle, im_name, im_path)
    hor_flip_and_save(img_arr, angle, im_name, im_path)
    ver_flip_and_save(img_arr, angle, im_name, im_path)
    ch_brightness_and_save(img_arr, angle, im_name, im_path)
    
def rotate_and_save(img_arr, angle, im_name, im_path):
    save_path = "\\".join(im_path.split("\\")[:-1]) + "\\r_" + str(angle) + im_name
    temp_rotation = rotate(img_arr, angle, resize=True)
    temp_rotation = normalize_img(temp_rotation)
    imsave(save_path, temp_rotation)
    return temp_rotation

def hor_flip_and_save(img_arr, angle, im_name, im_path):
    save_path = "\\".join(im_path.split("\\")[:-1]) + "\\hf_" + str(angle) + im_name
    hor_img = img_arr[:, ::-1]
    hor_img = normalize_img(hor_img)
    imsave(save_path, hor_img)
    ver_flip_and_save(hor_img, angle, im_name, im_path, True)
    ch_brightness_and_save(hor_img, angle, im_name, im_path, 1)
    

def ver_flip_and_save(img_arr, angle, im_name, im_path, from_hf= False):
    ver_img = img_arr[::-1, :]
    
    if not from_hf:
        save_path = "\\".join(im_path.split("\\")[:-1]) + "\\vf_" + str(angle) + im_name
        ch_brightness_and_save(ver_img, angle, im_name, im_path, 2)
    else:
        save_path = "\\".join(im_path.split("\\")[:-1]) + "\\hvf_" + str(angle) + im_name
        ch_brightness_and_save(ver_img, angle, im_name, im_path, 3)
        
    ver_img = normalize_img(ver_img)
    imsave(save_path, ver_img)


def ch_brightness_and_save(img_arr, angle, im_name, im_path, func_call = 0):
    for br in np.arange(0.4, 1.7, 0.4):
        if func_call == 1:
            save_path = "\\".join(im_path.split("\\")[:-1]) + "\\hfchbr_" + str(angle) + "_" +str(round(br, 1)) + im_name
        elif func_call == 2:
            save_path = "\\".join(im_path.split("\\")[:-1]) + "\\vfchbr_" + str(angle) + "_" +str(round(br, 1)) + im_name
        elif func_call == 3:
            save_path = "\\".join(im_path.split("\\")[:-1]) + "\\hvfchbr_" + str(angle) + "_" +str(round(br, 1)) + im_name
        else:
            save_path = "\\".join(im_path.split("\\")[:-1]) + "\\chbr_" + str(angle) + "_" +str(round(br, 1)) + im_name
        temp_img = exposure.adjust_gamma(img_arr, gamma=br, gain=1)
        temp_img = normalize_img(temp_img)
        imsave(save_path, temp_img)
        
def normalize_img(img):
    return cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def augmentation(img_path):
    im = imread(img_path)
    im_name = img_path.split("\\")[-1]
    for angle in range(30, 360, 30):
        augment_pipeline(im, angle, im_name, img_path)
    return "Done Processing"


def main():
    start_time = time.time()
    max_workers = mp.cpu_count() - 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        res = [executor.submit(augmentation, img_path) for img_path in images]

    for f in concurrent.futures.as_completed(res):
        print(f.result())

    print("total time", time.time()- start_time)

    
if __name__ == '__main__':
    main()
