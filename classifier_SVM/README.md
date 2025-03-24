# PCA + SVM 
This pipeline of model conducts binary classification on spectra from Sherwood simulation suite of redshift 2.4. The classes are 'nofeedback' and 'strongAGN'. 

To conduct the training:
First adjust the hyperparameters in config.py.
Then run "python main.py" in terminal.

# Results

* Conducted training with PCA+SVM
* Total data size: 3000 per class (6000 spectra total)
* Key findings:
  * PCA did not improve performance
  * Models with PCA tended to predict 'nofeedback' majority of the time
  * Removing PCA from the pipeline resolved the majority class prediction issue
  * Accuracy remained similar (~60%) without PCA
  * Doubling total data size reduced overfitting but hardly improved accuracy
* Best hyperparameter combinations:
  *{'C': 0.01, 'gamma': 0.1, 'kernel': 'poly'}
  *{'C': 0.05, 'gamma':'scale', 'kernel':'rbf'}



