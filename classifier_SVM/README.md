# PCA + SVM 
This pipeline of model conducts binary classification on spectra from Sherwood simulation suite of redshift 2.4. The classes are 'nofeedback' and 'strongAGN'. 

To conduct the training:
First adjust the hyperparameters in config.py
Then run "python main.py" in terminal.



Quick note on the result:
Conducted training with PCA+SVM 
Total data size 3000 per class, total 6000 spectra
Conducting PCA didn't help. Models ended up predicting spectra as 'nofeedback' majority of the time. 
Not utilizing PCA solved this issue, but the accuracy didn't improve.
Doubling the total data size reduced overfitting, but accuracy remained similar around 60%. 
Using parameter grid 3, the best hyperparameter combination was {'C': 0.01, 'gamma': 0.1, 'kernel': 'poly'}. 


