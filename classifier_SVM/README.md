# PCA + SVM 
This pipeline of model conducts binary classification on spectra from Sherwood simulation suite of redshift 2.4. The classes are 'nofeedback' and 'strongAGN'. 

To conduct the training:
First adjust the hyperparameters in config.py.
Then run "python main.py" in terminal.

# Results and Thoughts

* Conducted training with PCA+SVM
* Total data size: 3000 per class (6000 spectra total)
* Key findings:
  * PCA did not improve performance
  * Models with PCA tended to predict 'nofeedback' majority of the time
  * Removing PCA from the pipeline resolved the majority class prediction issue
  * Accuracy remained similar (~60%) without PCA
  * Doubling total data size reduced overfitting but hardly improved accuracy
* Best hyperparameter combinations in terms of accuracy:
  *{'C': 0.01, 'gamma': 0.1, 'kernel': 'poly'}
  *{'C': 0.05, 'gamma':'scale', 'kernel':'rbf'}
  (Without PCA)
* Confusion Matrices 

  *ncomp194, grid3
  {'C': 0.01, 'gamma': 0.1, 'kernel': 'poly'}
                precision    recall  f1-score   support
    
  0(nofeedback)      0.61      0.70      0.65      1200
  1(strongAGN)       0.65      0.55      0.59      1200

      accuracy                           0.62      2400
    macro avg       0.63      0.62      0.62      2400
  weighted avg       0.63      0.62      0.62      2400

  Precision = TP / (TP+FP),  
  Recall = TP / (TP+FN) 

  When the model predicts 'nofeedback', 61% of the time it is correct

  70% of the 'nofeedback' data points were correctly predicted. 

  55% of the 'strongAGN' data points were correctly predicted.


  *ncomp194, grid4
  {'C': 0.05, 'gamma':'scale', 'kernel':'rbf'}
                precision    recall  f1-score   support

  0(nofeedback)      0.58      0.88      0.70      1200
  1(strongAGN)       0.75      0.36      0.49      1200

      accuracy                           0.62      2400
    macro avg       0.67      0.62      0.59      2400
  weighted avg       0.67      0.62      0.59      2400

  88% percent of the 'nofeedback' data points were correctly predicted, while only 36% of the 'strongAGN' data points were correctly predicted. 

  This means the model is heavily biased toward 'nofeedback' class, and not good at detecting 'strongAGN' class. 

  In fact, applying PCA for dimensionality reduction before classification worsened this bias. For example with 12 principal components, 

  ncomp12, grid3
              precision    recall  f1-score   support

           0       0.55      0.98      0.70       600
           1       0.93      0.19      0.31       600

    accuracy                           0.59      1200
   macro avg       0.74      0.59      0.51      1200
  weighted avg       0.74      0.59      0.51      1200



