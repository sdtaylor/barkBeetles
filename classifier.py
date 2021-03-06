from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import gdalnumeric
import warnings
import sklearn
import pandas as pd
import numpy as np
import gdal
from gdalconst import *
warnings.filterwarnings('ignore')


testingFolder='./data/testingArea/'
#################################################################
#Read in data written extractData.py
data=pd.read_csv('bbCleanedData.csv')

y=data['t1'].values
X=data.drop(['t1'], axis=1)

#Death bins corrospond to transformed data. 
treeDeathBins=np.array([0,1,2,3,4,5,6,7,8,9,10])

#Make the t1 data into dummy variables instead of a single column
t1_catagories=['t1_is_'+str(i) for i in np.sort(X['t'].unique())]
encoder=LabelBinarizer()
encoded_catagories=encoder.fit_transform(X['t'])

for i,label in enumerate(t1_catagories):
    X[label]=encoded_catagories[:,i]
X.drop('t',1, inplace=True)

#Feature names for graphs and stuff
X_feature_names=X.columns.values

######################################################################################
#Prepare tree cover data.
treeCover=gdalnumeric.LoadFile(testingFolder+'tree_cover.tif')
treeCoverBins=np.array([0,10,20,30,40,50,60,70,80,90,110])

#Pad array with no trees as "edges"
treeCover=np.vstack((treeCover, np.zeros(treeCover.shape[1])))
treeCover=np.vstack((np.zeros(treeCover.shape[1]), treeCover))
treeCover=np.hstack((treeCover, np.zeros(treeCover.shape[0]).reshape((treeCover.shape[0],1))))
treeCover=np.hstack((np.zeros(treeCover.shape[0]).reshape((treeCover.shape[0],1)), treeCover))

#################################################################
#Tune the decision tree hyperparamters. This tunes the decision tree parameters using a particle swarm
#optimizaion that minimizes the log loss of the classification. It takes about 20 minutes to run on a
#hipergator server with 50 cores, so don't do it on your laptop.

#import optunity
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss

def optimize_tree_parameters():
    #Split the data up into 75% training and 25% testing size, stratified over all classes.
    ss=StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=1)
    train, test = next(iter(ss))
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    #function for optunity package to minimize.
    def tree(max_depth=None, min_samples_leaf=None, min_samples_split=None, max_features=None):
        clf=DecisionTreeClassifier(random_state=1,
                                   max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf),
                                   min_samples_split=int(min_samples_split), max_features=max_features)
        clf.fit(X_train, y_train)
        predictions=clf.predict_proba(X_test)
        return(log_loss(y_test, predictions))

    #Parameters to tune. this matches up with the inputs to DecisionTreeClassifer inside tree()
    params={'max_depth':[2,20],
            'min_samples_leaf':[1,100],
            'min_samples_split':[2,50],
            'max_features':[0,1]}

    print('optimizing')
    #Set number of cores to use here.
    p=optunity.parallel.create_pmap(50)
    optimal_params, extra_info, solver_info = optunity.minimize(tree, num_evals=10000, pmap=p, **params)

    print(optimal_params)

#Uncomment to get optimzed parameters. Also need to uncomment 'import optunity' above
#optimize_tree_parameters()
#output from runing this. 
optimized_params={'max_features': 0.36, 'min_samples_leaf': 70, 'max_depth': 10, 'min_samples_split': 27}

######################################################################################
#Write out a raster from a numpy array.
#Template: a raster file on disk to use for pixel size, height/width, and spatial reference.
#array: array to write out. Should be an exact match in height/width as the template.
#filename: file name of new raster
#inspired by: http://geoexamples.blogspot.com/2012/12/raster-calculations-with-gdal-and-numpy.html
def write_array(template_object, array, filename):
    driver = gdal.GetDriverByName("GTiff")
    raster_out = driver.Create(filename, template_object.RasterXSize, template_object.RasterYSize, 1, template_object.GetRasterBand(1).DataType)
    gdalnumeric.CopyDatasetInfo(template_object,raster_out)
    bandOut=raster_out.GetRasterBand(1)
    gdalnumeric.BandWriteArray(bandOut, array)

####################################################################################
#Takes an array from a certain timestep and retuns the data in a format that can be
#put into the decision tree classifier
def extract_data(array):
    #Pad array with 0 tree deaths as "edges"
    array=np.vstack((array, np.zeros(array.shape[1])))
    array=np.vstack((np.zeros(array.shape[1]), array))
    array=np.hstack((array, np.zeros(array.shape[0]).reshape((array.shape[0],1))))
    array=np.hstack((np.zeros(array.shape[0]).reshape((array.shape[0],1)), array))

    allData=[]
    for row in range(1, array.shape[0]-1):
        for col in range(1, array.shape[1]-1):
            thisPixelData={}

            thisPixelData['treeCover']=treeCover[row,col]
            thisPixelData['t']=array[row,col]
            surrounding=array[row-1:row+2, col-1:col+2].reshape((9))
            surrounding=np.delete(surrounding, 4)

            surroundingSize=len(surrounding)
            for catagory in range(1, len(treeDeathBins)+1):
                thisPixelData['Surrounding-Cat'+str(catagory)]= np.sum(surrounding==catagory) / surroundingSize

            allData.append(thisPixelData)


    allData=pd.DataFrame(allData)
    #Convert t catagories to dummy labels
    encoded_catagories=encoder.transform(allData['t'])
    for i,label in enumerate(t1_catagories):
        allData[label]=encoded_catagories[:,i]
    allData.drop('t',1, inplace=True)

    return(allData)

######################################################################################
#The fitted classifier on the full data set.
def model_object(X, y, **model_params):
    model=DecisionTreeClassifier(random_state=1, **model_params)
    model.fit(X,y)
    return(model)


######################################################################################
#Create the diagram of the decision tree.
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from os import system
def create_tree_diagram(model, feature_names):
    dot_data = StringIO()
    export_graphviz(model, out_file='modelTree.dot',
                    feature_names=X_feature_names,
                    class_names=['Class-'+str(i) for i in treeDeathBins[0:-1]])
                    #class_names=['0', '100', '250', '500', '750', '1000', '1500', '2000', '2500', '5000'])
    system('dot -T png modelTree.dot -o modelTree.png')

#####################################################################################
#The stochastic part of the model. 
#Randomly choose a given class given probabilites for each class
def stochastic_predict(prob_matrix, classes):
    pred=np.zeros(prob_matrix.shape[0])
    #For each observation (row). independently and randomly choose a class based on the probabilites
    for row in range(prob_matrix.shape[0]):
        pred[row] = np.random.choice(classes, p=prob_matrix[row,])
    return(pred)


#####################################################################################
#Draw side by side image of actual and prediction for all years
def draw_side_by_side(actual, prediction, years):
    actual=np.digitize(np.log1p(actual), treeDeathBins)

    fig=plt.figure(figsize=(16,22))
    n=1
    for i, year in enumerate(years):
        plt.subplot(5,2,n)
        plt.imshow(actual[:,:,i], cmap=plt.get_cmap('Reds'), vmax=np.max(full_model.classes_), vmin=np.min(full_model.classes_))
        plt.title(str(year)+' Actual')
        n+=1
        plt.subplot(5,2,n)
        plt.imshow(prediction[:,:,i], cmap=plt.get_cmap('Reds'), vmax=np.max(full_model.classes_), vmin=np.min(full_model.classes_))
        plt.title(str(year)+' Prediction')
        n+=1
    plt.tight_layout()
    plt.show()


#####################################################################################
#Write out geo referenced rasters of all actual maps and predictions
def write_all_rasters(actual, prediction, years, template):
    actual=np.digitize(np.log1p(actual), treeDeathBins)
    for i, year in enumerate(years):
        write_array(template, prediction[:,:,i], './results/mpb_prediction_'+str(year)+'.tif')
        write_array(template, actual[:,:,i], './results/mpb_actual_'+str(year)+'.tif')

#####################################################################################
#Get kappa values for each year
from sklearn.metrics import cohen_kappa_score
def get_kappas(actual, prediction, years):
    actual=np.digitize(np.log1p(actual), treeDeathBins)
    size=actual.shape[0]*actual.shape[1]
    for i_year, year in enumerate(years):
        print(year, cohen_kappa_score(actual[:,:,i_year].reshape(size), prediction[:,:,i_year].reshape(size)))


#####################################################################################
#Calculate the percentages of all classes within each actual and predicted images in each year
#Pack them into a dataframe for export to csv
def get_percentages(actual, prediction, years):
    actual=np.digitize(np.log1p(actual), treeDeathBins)

    total_pixels=actual.shape[0]*actual.shape[1]

    df=[]
    for i_year, year in enumerate(years):
        predicted_pct=np.bincount(prediction[:,:,i_year].reshape((prediction.shape[0]*prediction.shape[1])).astype(int))[1:]
        actual_pct=np.bincount(actual[:,:,i_year].reshape((prediction.shape[0]*prediction.shape[1])).astype(int))[1:]

        predicted_pct = predicted_pct / total_pixels
        actual_pct = actual_pct / total_pixels

        #Put in 0's for higher catagories if they are not in the training or predicted data
        #This happens sometimes because of the stochastic nature.
        while len(treeDeathBins[:-1]) > len(actual_pct):
            actual_pct = np.append(actual_pct, 0)
        while len(treeDeathBins[:-1]) > len(predicted_pct):
            predicted_pct = np.append(predicted_pct, 0)

        for i_class, this_class in enumerate(treeDeathBins[:-1]):
            catagory_name=str(np.rint(np.expm1(this_class)))+' - '+str(np.rint(np.expm1(treeDeathBins[i_class+1])))
            if i_class == len(treeDeathBins)-2:
                catagory_name=str(np.rint(np.expm1(this_class)))+'+'

            df.append({'year':year, 'catagory':catagory_name, 'pct':actual_pct[i_class], 'type':'actual'})
            df.append({'year':year, 'catagory':catagory_name, 'pct':predicted_pct[i_class], 'type':'predicted'})

    #These are 2d arrays (cols: classes * rows: years), that give the percentage of pixels in each class in each year.
    return(pd.DataFrame(df))

#####################################################################################
#Start doing stuff here. 


full_model=model_object(X,y, **optimized_params)

#Creating the tree diagram requires the graphviz program.
#create_tree_diagram(full_model, X_feature_names)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
#The width, height, CRS, and pixel size of the template will be
#used to write rasters that were modified using numpy arrays
template=gdal.Open(testingFolder+'tree_cover.tif', GA_ReadOnly)

#Inital conditions from 2005
prediction=np.digitize(np.log1p(gdalnumeric.LoadFile(testingFolder+'mpb_2005.tif')), treeDeathBins)
area_shape=prediction.shape

last_year_actual=gdalnumeric.LoadFile(testingFolder+'mpb_2005.tif')

year_list=list(range(2006,2011))

all_years_predictions=np.zeros((area_shape[0], area_shape[1], len(year_list)))
all_years_actual=np.zeros((area_shape[0], area_shape[1], len(year_list)))


#Run the model for the 5 years. Also save the observed data as we go along. 
for i, year in enumerate(year_list):
    #Get probabilites for all cell values for t+1. Make stochastic predictions and 
    #convert them back into an array
    probabilites = full_model.predict_proba(extract_data(prediction))
    prediction = stochastic_predict(probabilites, full_model.classes_).reshape(area_shape)

    this_year_actual=gdalnumeric.LoadFile(testingFolder+'mpb_'+str(year)+'.tif') + last_year_actual
    last_year_actual=this_year_actual

    all_years_predictions[:,:,i]=prediction
    all_years_actual[:,:,i]=this_year_actual


#Make graphs and results files with those predictions.
draw_side_by_side(all_years_actual, all_years_predictions, year_list)
#write_all_rasters(all_years_actual, all_years_predictions, year_list, template)
print(get_kappas(all_years_actual, all_years_predictions, year_list))
pct_results=get_percentages(all_years_actual, all_years_predictions, year_list)
pct_results.to_csv('class_percentages.csv', index=False)
