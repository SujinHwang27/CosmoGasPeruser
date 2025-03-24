# PCA + SVM 
This pipeline of model conducts binary classification on spectra from Sherwood simulation suite of redshift 2.4. The classes are 'nofeedback' and 'strongAGN'. 

To conduct the training:
First adjust the hyperparameters in config.py.
Then run "python main.py" in terminal.

# Results and Thoughts

* Conducted model development with PCA+SVM for binary classification of the synthetic quasar absorption spectra. Classes are labeled as 0 (no feedback universe) and 1 (strong AGN feedback universe). Initially started with total 6000 spectra, 3000 per class.
* Key findings:
  Applying PCA for dimensionality reducsion did not improve performance. Models with PCA tended to predict 'nofeedback' majority of the time. Removing PCA from the pipeline resolved the majority class prediction issue, but didn't improve the accuracy, which remained around 62%. Doubling total data size solved overfitting issue, but hardly improved accuracy. 
* Best hyperparameter combinations in terms of accuracy found without PCA:
  
  {'C': 0.01, 'gamma': 0.1, 'kernel': 'poly'} and {'C': 0.05, 'gamma':'scale', 'kernel':'rbf'}
  
* Confusion Matrices comparison
  
  (Precision = TP / (TP+FP),  Recall = TP / (TP+FN). Reading precision in the first table, for example, can be: 'when the model predicts 'nofeedback', 61% of the time it is correct.' However, reading recall seems to be more adequate in this problem. )

  * {'C': 0.01, 'gamma': 0.1, 'kernel': 'poly'}
    
  | Class            | Precision | Recall | F1-Score | Support |
  |-----------------|-----------|--------|----------|---------|
  | 0 (nofeedback)  | 0.61      | 0.70   | 0.65     | 1200    |
  | 1 (strongAGN)   | 0.65      | 0.55   | 0.59     | 1200    |
  | **Accuracy**    |           |        | 0.62     | 2400    |
  | **Macro Avg**   | 0.63      | 0.62   | 0.62     | 2400    |
  | **Weighted Avg**| 0.63      | 0.62   | 0.62     | 2400    |

  * {'C': 0.05, 'gamma':'scale', 'kernel':'rbf'}
    
  | Class            | Precision | Recall | F1-Score | Support |
  |-----------------|-----------|--------|----------|---------|
  | 0 (nofeedback)  | 0.58      | 0.88   | 0.70     | 1200    |
  | 1 (strongAGN)   | 0.75      | 0.36   | 0.49     | 1200    |
  | **Accuracy**    |           |        | 0.62     | 2400    |
  | **Macro Avg**   | 0.67      | 0.62   | 0.59     | 2400    |
  | **Weighted Avg**| 0.67      | 0.62   | 0.59     | 2400    |

 With {'C': 0.01, 'gamma': 0.1, 'kernel': 'poly'}, 70% of the 'nofeedback' data points were correctly predicted and 55% of the 'strongAGN' data points were correctly predicted. With {'C': 0.05, 'gamma':'scale', 'kernel':'rbf'}, 88% percent of the 'nofeedback' data points were correctly predicted, while only 36% of the 'strongAGN' data points were correctly predicted. This means the model is heavily biased toward 'nofeedback' class, and not good at detecting 'strongAGN' class. In fact, applying PCA for dimensionality reduction before classification worsened this impbalance. For example, with 12 principal components, 

  | Class            | Precision | Recall | F1-Score | Support |
  |-----------------|-----------|--------|----------|---------|
  | 0 (nofeedback)  | 0.55      | 0.98   | 0.70     | 600     |
  | 1 (strongAGN)   | 0.93      | 0.19   | 0.31     | 600     |
  | **Accuracy**    |           |        | 0.59     | 1200    |
  | **Macro Avg**   | 0.74      | 0.59   | 0.51     | 1200    |
  | **Weighted Avg**| 0.74      | 0.59   | 0.51     | 1200    |

  it was shown that only 19% 'strongAGN' data points were predicted correctly. 


# Conclusion and Next Steps






