from skimage import color
from skimage import io
from skimage import feature
import cv2
import numpy as np
from PIL import Image
from skimage.feature import corner_harris, corner_peaks
from skimage.draw import circle

from skimage import measure

from numpy.linalg import norm
from copy import deepcopy

from tkinter import Tk
from tkinter.filedialog import askopenfilename


def edgDetection(file_path):

    im = color.rgb2gray(io.imread(file_path))
   
    print("image size is:",im.size) 
    
    #up sample image based on image size to improve the resolution
    if im.size > 5000000:
        upsampleSize = 2
    else:
        upsampleSize = 4

    newX, newY = im.shape[1]*upsampleSize, im.shape[0]*upsampleSize
    newImg = cv2.resize(im, (int(newX), int(newY)))
    print("image is resized")

    edges = feature.canny(newImg, sigma=1)#im, sigma=8)
    print("canny is done.")

    # number of points in the array return by canny filter
    nPoints = np.sum(edges)
    pts = np.zeros((nPoints, 2))

    m = 0
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j] == True:
                pts[m, 0] = i
                pts[m, 1] = j
                m += 1

    interpolImg = np.zeros((edges.shape[0], edges.shape[1]), int)
    print("interpolation is done.")
    for i in range(pts.shape[0]):
        interpolImg[int(pts[i, 0]), int(pts[i, 1])] = 255

    return interpolImg, pts

def sortPntsBasedOnImg(contour, crnPnts):
    srtPnts = []
    crnPnts = crnPnts.tolist()
    for [xc,yc] in contour:
       
        for [xp,yp] in crnPnts:
          for i in range(4): 
            if [xc+i,yc+i] == [xp,yp] or [xc+i,yc]==[xp,yp] or [xc,yc+i]==[xp,yp] or [xc-i,yc]==[xp,yp] or [xc,yc-i]==[xp,yp] or [xc-i,yc-i]==[xp,yp] or [xc+i,yc-i]==[xp,yp] or [xc-i,yc+i]==[xp,yp]:
               srtPnts.append([xp,yp])
               crnPnts.remove([xp,yp])
          
    return srtPnts


if __name__ == "__main__":
     
    root = Tk()
    Tk().withdraw()
    root.withdraw()     
    file_path = askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpg files","*.jpg"),("jpeg files","*.jpeg"),("png files","*.png")))
    root.update()

    if file_path == "":
        exit()
    
    edgImg, edgPnts = edgDetection(file_path)
 
    contours = measure.find_contours(edgImg/255, 0)

    corners = corner_harris(edgImg,2,3,0.0001)
    coords = corner_peaks(corners, min_distance=10)#50)

    srtPnts = []
    for i in range(len(contours)):
        m = divmod(i, 2)
        if m[1] == 0:
            srtPnts.append(sortPntsBasedOnImg(contours[i], coords))
        srtPnts.append(sortPntsBasedOnImg(contours[len(contours)-1], coords))
  
    for j in range(len(srtPnts)):
      for i in range(len(srtPnts[j])):
        rr, cc = circle(srtPnts[j][i][0], srtPnts[j][i][1], 5)
        edgImg[rr, cc] = 255 

    #invert image (black->white and  white->black)
    edgImg = np.invert(edgImg) 
    edgImg = (255-edgImg) 

    Image.fromarray(np.asarray(edgImg, dtype=np.int8), 'L').show()
    
    io.imsave('result', edgImg)



