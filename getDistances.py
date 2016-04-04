import numpy as np
import gdalnumeric

treeDeathBins=np.array([0, 100, 250,500,750,1000,1500,2000,2500,5000,50000])

image=gdalnumeric.LoadFile('./data/mpb_2010.tif')
image=np.digitize(image, treeDeathBins)
classes=np.unique(image)

class_locations=[]
max_classes=0
for this_class in classes:
    class_locations.append(np.argwhere(np.abs(image-this_class)==0))

pixel=0
for row in range(image.shape[0]):
    for col in range(image.shape[1]):
        for i, this_class in enumerate(classes):
            print(pixel)
            pixel+=1
            locations=class_locations[i]
            x=np.min((((row-locations[:,0])**2)+((col-locations[:,1])**2))**0.5)  #distance array to single point toCheck
