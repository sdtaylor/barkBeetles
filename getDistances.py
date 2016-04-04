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
    if class_locations[-1].shape[0]>max_classes:
        max_classes=class_locations[-1].shape[0]

locations=np.zeros((max_classes, 2, len(classes)))
locations.fill(-99999)

for i, array in enumerate(class_locations):
    locations[0:array.shape[0],:,i]=array


all_locations=np.argwhere(image > -1)
x=np.min((((all_locations[:,0]-locations[:,0,:]**2)+((all_locations[:,1]-locations[:,1,:])**2))**0.5), 1)
print(x.shape)
exit()

pixel=0
for row in range(image.shape[0]):
    for col in range(image.shape[1]):
        for
        print(pixel)
        pixel+=1
        x=np.min((((row-locations[:,0,:])**2)+((col-locations[:,1,:])**2))**0.5, 0)  #distance array to single point toCheck
