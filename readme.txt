Contents:

1. data folder
    Holds the data for 2 years, western MT and northern CO, testing and training respecitively. Each has tif files for raw dead tree numbers from 1997-2010 as well as a tif for % tree cover.

2. extractData.py
    this extracts all the data from the training area and writes it to the file bbCleanedData.csv

3. classifier.py
    this builds a decision tree model using data from bbCleanedData.csv, and applies it to the rasters in data/trainingArea/
    Outputs are:
        class_percentages.csv: landscape percentage for each class, for each year, in the model prediction and in the observations
        modelTree.png: a diagram of the decision tree (requires graphviz installed)
        printed kappa values: yearly kappa values for prediction vs observed
        printed side by side maps: pictures of the training area model predictions and observations for each year

4. percentage_graph.R
    this takes class_percentages.csv and creates bar charts to compare the year to year composition of each catagory in the
    predictions vs observations
    Outputs:
        training_area_bar_plot.jpg

5. cutToStudyArea.R
    takes the raw data from Meddens et al (not included here) and crops to my training and testing area

6. getDistances.py
    experiment for including, for each pixel, the distance to the nearest catagory i. ended up not using this. 

--------------------------------------
Required packages
Python:
    numpy, pandas, gdal, optunity (not needed to repeat analysis below), sklearn

    if using anaconda, you just need to install gdal and optunity with the conda installer. 

R:
    dplyr, ggplot2, RColorBrewer

-------------------------------------
To repeat analysis:
1. run extractData.py
2. run classifier.py (prints some things to stdout and writes some results files)
3. run percentage_graph.R to make the bar charts. 
