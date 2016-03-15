import numpy as np
import gdalnumeric



dataDir='./data/'

#One mountain pine beetle map for each ear
bbFiles=[dataDir+'mpb_'+str(year)+'.tif' for year in range(1997,2011)]

def stackImages(fileList):
    #Read in all rasters and stack them into a single array. (rows x column x numRasters)
    fullYear=gdalnumeric.LoadFile(fileList[0])
    for thisImage in fileList[1:]:
        image=gdalnumeric.LoadFile(thisImage)
        fullYear=np.dstack((fullYear, image))
    return(fullYear)

imageStack=stackImages(bbFiles)

#extract an arbitrary pixel (x,y) value at time t, t+1, and it's surrounding n pixel values at time t
def extractValue(imageStack,x,y,t, n):
    focalPixel_t=imageStack[x,y,t]
    focalPixel_t1=imageStack[x,y,t+1]
    
    if n==8:
        surrounding=imageStack[x-1:x+2 , y-1:y+2, t].reshape((9))
    elif n==4:
        surrounding=imageStack[x-1:x+2 , y-1:y+2, t].reshape((9))

        



    pass
#
