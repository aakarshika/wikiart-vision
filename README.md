### Genrewise Classification of Fine Art Paintings from Wikiart


Bhavika Tekwani, Akarshika Priydarshi

## Prerequisites:

Anaconda 3.4 or higher. 

## Environment:

1) Clone this repository.
2) Create a conda environment and give it a name (say 'testenv')

`` conda create -n testenv python=3.4 anaconda ``

3) Install all the packages from the requirements.txt file:

`` while read requirement; do conda install --yes $requirement; done < requirements.txt ``

4) Find instructions to install OpenCV-contrib on your system here:

https://www.pyimagesearch.com/opencv-tutorials-resources-guides/

5) Instructions to install PyLearGist on Python 3+ are in Notes/leargist_setup.md.

6) Make a .env file in your project folder which has all the constants and paths being used
in the Python scripts. There is an example called sample.env provided in this ZIP file.

# Files

These are described in the order in which they must be run.
Each step may take very long to run which is why there is no single command to run all the experiments.

There are other files which are not mentioned in this list because they don't have to
be run to replicate the results, they are mostly utlity files to generate visualizations, 
evaluations or data. 

1) src/create_dataset.py: Generates the dataset based on how many classes & images are needed.
2) src/feature_extraction.py: Creates Numpy arrays which have all the features for train and test datasets.
These feature arrays will be stored in the '/data' folder under your project.
Additionally, it also creates vocabulary files for train and test and arrays containing GIST descriptors.
3) src/models.py: Contains ML models like kNN, Random Forest and XGBoost along with the GridSearchCV pipeline.
4) src/resnet.py: The entire ResNet18 for end-to-end data loading, training, testing and model evaluation steps.
5) src/cnn.py: Dor end-to-end data loading, training, testing and model evaluation of the 2 Layer CNN. 
6) src/resnet_eval.py: Assuming you have a folder called 'models' in your project, this file will automatically look
for stored prediction pickles for each type of CNN and ResNet with varying learning rates and calculate various 
performance metrics for the model. 


# Data

1) To download the Wikiart dataset, go (here)[https://github.com/cs-chan/ICIP2016-PC/tree/master/WikiArt%20Dataset].
The entire dataset is 27 GB. 
2) /data: This folder must be created by you under the project directory so that
other scripts can store feature numpy arrays here. 
3) /models: All model checkpoints for ResNet18 and CNNs are stored here. Prediction 
and actual pickled lists are also saved here by the resnet.py and cnn.py scripts.


# Other files:

1) aws.env: An environment file to manage my EC2 instance on AWS.
2) requirements.txt: Contains all the dependencies to run this codebase. 
3) wikiart.py: The DataLoader for managing our CNN training. 
4) viz/confusion_matrix.py: A utility script to generate the confusion matrix for the 
ResNet18 model we have included in the paper. 