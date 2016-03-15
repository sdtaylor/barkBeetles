#Take the original exports of the full western US and cut them to the study area in north central coloardo.
#also add together the rasters for lodgepole pine and ponderosa pine to get total tree deaths/pixel
library(raster)
library(rgdal)
library(sp)

rasterLocation='~/data/barkbeetle/exported_tifs/'
newLocation='~/data/barkbeetle/study_area_tifs/'

fileList=list.files(rasterLocation, pattern='*tif', full.names = TRUE)

fileList=fileList[!grepl('xml', fileList)]
fileList=fileList[!grepl('ovr', fileList)]

#studyAreaOutline=readShapePoly('/home/shawn/projects/barkBeetle/gisData/studyAreaOutline.shp', proj4string = CRS(utm_Z12_CRS))
studyAreaOutline=readOGR('/home/shawn/projects/barkBeetle/gisData', 'studyAreaOutline')
studyAreaOutline=spTransform(studyAreaOutline, CRS('+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs +ellps=GRS80 +towgs84=0,0,0 '))

#Rasters have -9999 values for NA that need to be dealt with before adding together the different tree species.
addRasters=function(x,y){
  xNULL=x<0
  yNULL=y<0
  
  x[xNULL]=0
  y[yNULL]=0
  
  return(x+y)
}

for(thisYear in 1997:2010){
  files=grep(thisYear, fileList, value=TRUE)
  stacked=raster::stack(files)
  stacked=crop(stacked, studyAreaOutline)
  
  
  addedTogether=overlay(stacked, fun=addRasters
  
  newPath=paste(newLocation,'mpb_',thisYear,'.tif',sep='')
  writeRaster(addedTogether, newPath)
  
}
