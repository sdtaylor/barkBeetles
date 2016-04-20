import numpy as np
import pandas as pd
import gdalnumeric


#Read in all rasters and stack them into a single array. (rows x column x numRasters)
def stackImages(fileList):
    fullYear=gdalnumeric.LoadFile(fileList[0])
    for thisImage in fileList[1:]:
        image=gdalnumeric.LoadFile(thisImage)
        fullYear=np.dstack((fullYear, image))
    return(fullYear)

#extract an arbitrary pixel (x,y) value at time t, t+1, and it's surrounding n pixel values at time t
def extractValue(stack,x,y,t, n):
    focalPixel_t=stack[x,y,t]
    focalPixel_t1=stack[x,y,t+1]

    if n==8:
        surrounding=stack[x-1:x+2 , y-1:y+2, t].reshape((9))
        #Delete the focal pixel that is in this 3x3 array
        surrounding=np.delete(surrounding, 4)
    elif n==24:
        surrounding=imageStack[x-2:x+3 , y-2:y+3, t].reshape((25))
        surrounding=np.delete(surrounding, 12)

    return(focalPixel_t, focalPixel_t1, surrounding)

###################################################################
###################################################################
dataDir='./data/trainingArea/'

#dead tree data. one mountain pine beetle map per year
bbFiles=[dataDir+'mpb_'+str(year)+'.tif' for year in range(2005,2011)]

imageStack=stackImages(bbFiles)

#number of dead trees can decrease over time because surveyers technically only count
#dieing trees (beetle kill red stage), and not trees several years dead. For the model I'll make it
#so the number of dead at any time period = sum of the dead trees in all prior years
for time in range(1,imageStack.shape[2]):
    #Value of each pixel = sum of that year plus the prior year
    imageStack[:,:,time]=imageStack[:,:,time-1:time+1].sum(axis=2)


treeDeathBins=np.array([0,1,2,3,4,5,6,7,8,9,10])

imageStack=np.digitize(np.log1p(imageStack), treeDeathBins)


#The tree cover base layer. Gives percent cover of trees and makes it so the model doesn't spread
#beetles where there are no trees. It's binned just like the trees into catagorical classes of tree cover
treeCover=gdalnumeric.LoadFile('./data/trainingArea/tree_cover.tif')


#Extract all values into a format suitable for sklearn. 
count=0
data=[]
for row in range(1, imageStack.shape[0]-1):
    for col in range(1, imageStack.shape[1]-1):
        #First get the base tree cover for this pixel
        thisPixelTreeCover=treeCover[row,col]

        for time in range(0, imageStack.shape[2]-1):
            dataThisObs={}
            t, t1, surrounding=extractValue(imageStack, row, col, time, 8)

            dataThisObs['t']=t
            dataThisObs['t1']=t1
            dataThisObs['treeCover']=thisPixelTreeCover

            #Process the surrounding pixel data as fraction in each of the tree death number catagories
            surroundingSize=len(surrounding)
            for catagory in range(1, len(treeDeathBins)+1):
                dataThisObs['Surrounding-Cat'+str(catagory)]= np.sum(surrounding==catagory) / surroundingSize

            data.append(dataThisObs)

data=pd.DataFrame(data)
data.to_csv('bbCleanedData.csv', index=False)
