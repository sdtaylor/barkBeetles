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

#np.random.seed(1)

#################################################################
#Read in data written extractData.py
data=pd.read_csv('bbCleanedData.csv')

y=data['t1'].values
X=data.drop(['t1'], axis=1)
X_feature_names=data.drop(['t1'], axis=1).columns.values

treeDeathBins=np.array([0, 100, 250,500,750,1000,1500,2000,2500,5000,50000])

t1_catagories=['t1_is_'+str(i) for i in np.sort(X['t'].unique())]
encoder=LabelBinarizer()
encoded_catagories=encoder.fit_transform(X['t'])

for i,label in enumerate(t1_catagories):
    X[label]=encoded_catagories[:,i]

X.drop('t',1, inplace=True)

#################################################################
#Tune the decision tree hyperparamters. This tunes the decision tree parameters using a particle swarm
#optimizaion that minimizes the log loss of the classification. It takes about 20 minutes to run on a
#hipergator server with 50 cores, so don't do it on your laptop.
import optunity
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

#optimize_tree_parameters()
#output from runing this. 
optimized_params={'max_features': 0.68, 'min_samples_leaf': 91, 'max_depth': 9, 'min_samples_split': 10}
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

######################################################################################
#Extract all cell values and their surrounding values, along with non-changing tree cover data
treeCover=gdalnumeric.LoadFile('./data/tree_cover.tif')
treeCoverBins=np.array([0, 25, 50, 75, 110])
treeCover=np.digitize(treeCover, treeCoverBins)

#Pad array with no trees as "edges"
treeCover=np.vstack((treeCover, np.zeros(treeCover.shape[1])))
treeCover=np.vstack((np.zeros(treeCover.shape[1]), treeCover))
treeCover=np.hstack((treeCover, np.zeros(treeCover.shape[0]).reshape((treeCover.shape[0],1))))
treeCover=np.hstack((np.zeros(treeCover.shape[0]).reshape((treeCover.shape[0],1)), treeCover))

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
#take a list of values and put them back into an array
def insert_data(array1d, rows, cols):
    array2d=np.empty(rows*cols).reshape((rows, cols))


######################################################################################
#The fitted classifier on the full data set.
def model_object(X, y, **model_params):
    model=DecisionTreeClassifier(random_state=1, **model_params)
    #model=RandomForestClassifier(random_state=1)
    model.fit(X,y)
    return(model)


######################################################################################
#Create the diagram of the decision tree.
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from os import system
def create_tree_diagram(model, feature_names):
    dot_data = StringIO()
    export_graphviz(model, out_file='modelTree.dot')
                    #feature_names=X_feature_names)
                    #class_names=[str(i) for i in treeDeathBins[0:-1]])
                    #class_names=['0', '100', '250', '500', '750', '1000', '1500', '2000', '2500', '5000'])
    system('dot -T png modelTree.dot -o modelTree.png')

#####################################################################################
#Randomly choose a given class given probabilites for each class
def stochastic_predict(prob_matrix, classes):
    pred=np.zeros(prob_matrix.shape[0])
    #For each observation (row). independently and randomly choose a class based on the probabilites
    for row in range(prob_matrix.shape[0]):
        pred[row] = np.random.choice(classes, p=prob_matrix[row,])
    return(pred)


#####################################################################################
#Print cross classification report using 25% of data as hold out
def cross_validate(X,y, **model_params):
    ss=StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=1)
    #scores=cross_val_score(clf, X, y, scoring='f1', cv=ss)
    trainCV, testCV = next(iter(ss))
    X_train, X_test, y_train, y_test = X[trainCV], X[testCV], y[trainCV], y[testCV]

    model=DecisionTreeClassifier(random_state=1, **model_params)
    model.fit(X,y)
    y_pred=model.predict(X_test)
    print(classification_report(y_test, y_pred))






#print(cross_validate(X.values,y, **optimized_params))
full_model=model_object(X,y, **optimized_params)
#create_tree_diagram(full_model, X_feature_names)
#exit()

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
#The width, height, CRS, and pixel size of the template will be
#used to write rasters that were modified using numpy arrays
template=gdal.Open('./data/tree_cover.tif', GA_ReadOnly)

prediction=np.digitize(gdalnumeric.LoadFile('./data/mpb_2005.tif'), treeDeathBins)
area_shape=prediction.shape


fig=plt.figure()
n=1
for year in range(2006,2011):
    #prediction = full_model.predict(extract_data(prediction)).reshape(area_shape)
    probabilites = full_model.predict_proba(extract_data(prediction))
    prediction = stochastic_predict(probabilites, full_model.classes_).reshape(area_shape)

    #plt.imshow(prediction, cmap=plt.get_cmap('hot'), vmax=np.max(full_model.classes_), vmin=np.min(full_model.classes_))
    #plt.show()
    this_year_actual=np.digitize(gdalnumeric.LoadFile('./data/mpb_'+str(year)+'.tif'), treeDeathBins)

    plt.subplot(10,2,n)
    plt.imshow(this_year_actual, cmap=plt.get_cmap('hot'), vmax=np.max(full_model.classes_), vmin=np.min(full_model.classes_))
    plt.title(str(year)+' Actual')
    n+=1
    plt.subplot(10,2,n)
    plt.imshow(prediction, cmap=plt.get_cmap('hot'), vmax=np.max(full_model.classes_), vmin=np.min(full_model.classes_))
    plt.title(str(year)+' Prediction')
    n+=1
    #write_array(template, prediction, './results/mpb_prediction_'+str(year)+'.tif')
    #write_array(template, this_year_actual, './results/mpb_actual_'+str(year)+'.tif')
plt.show()
