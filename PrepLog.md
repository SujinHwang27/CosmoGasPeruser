This file is to track the processes of researching and preparing the project proposal \
Each section will take a part in the proposal document


# Introduction / Background
### 1. Main research questions to be addressed
Can we leverage machine learning techniques and simulated data to...\
  A. Predict physical properties of CGM? (Careful to say 'analyze')\
  B. Predict / backtrack Line-of-Sight (LOS) information?\



### 2. Goals for the project (explicitly outline the research motivation and my proposed work)
- Build and train machine learning model that can predict physical properties \
  such as density and temperature of the encountered CGMs from quasar absorption lines. 
- Leverage the cosmological simulations to obtain training data. \
  It is well known that quantity and the variety of training data is crucial to boost the performance of machine learning models. \
  Since observational data lacks in quantity and variety, synthetic data can be a good alternative. 



# Description of my previous work on the topic
- provide summary
- display the connection to the new topic


# Timeline for what the study will aim to accomplish within the grant award period (explicit detail)
- data preparation
- model selection (multiple) 
- hyperparameter tuning & training
- evaluation : need to come up with a reliable and reasonable metrics 
- publish open source (?)


# Description of how the study is relevant to NASA
(will be added)

## Doubts on the methodology of leveraging synthetic data and machine learning
- Simulations are not same as real universe : how much errror? 
- Synthetic data is different from real data : \
  This can be critical because the aim of the project is to build an algorithm that yields result that can be viewed as a near fact. \
  But if machine learning algorithms learn from synthetic facts, than how meaningful is it for real-life science? 
- Need a big assumption “synthetic data are similar to real data”.
  This can be dangerous because this study aims to create a scientific tool (or analyzer) that can help humanity get meaningful information from what is observed.
 	If the inference is not logically robust, then the result is meaningless.
- Error = Intrinsic error of the models + difference between the training and test data
  The former can't be improved. The latter need to be improved
- This project consists of assumptions. Error1 * Error2 * ... may end up being large 

## How to fix? 
- Ensure that synthetic data is similar to real data by...
  a. Getting similar distribution?
  b. Aligning the physical properties of the environment from which they were produced
- Be aware of the error of the machine learning algorithm ..?


## Literature Review
-	[Observing the circumgalactic medium of simulated galaxies through synthetic absorption spectra](https://academic.oup.com/mnras/article/479/2/1822/5046485)
 : physical properties of CGM (density, temperature, … ) through simulation
-	[Machine learning-based photometric classification of galaxies, quasars, emission-line galaxies, and stars](https://arxiv.org/pdf/2311.02951)
 : spectra classification with ml, comparison of different alg
-	[Mapping circumgalactic medium observations to theory using machine learning](https://academic.oup.com/mnras/article/525/1/1167/7241539#414139466)
 : The main objective of the study is to develop and apply a random forest (RF) framework to predict the physical conditions of the circumgalactic medium (CGM) from quasar absorption line observables. The study aims to bypass traditional simplifying assumptions in CGM modeling by using data from the simba cosmological simulation to better capture the complex relationships between CGM observables (such as H i and metal lines) and the underlying gas conditions (e.g., overdensities, temperatures, and metallicities). The models are trained to make accurate predictions of these physical conditions across various galaxy properties and absorbers, with a focus on improving predictive accuracy and understanding the feature importance that drives these predictions.
-	[Efficient identification of broad absorption line quasars using dimensionality reduction and machine learning](https://arxiv.org/abs/2404.12270)
	 : Quasar classification, absorption lines, PCA is the best
-	[An Interpretable Machine-learning Framework for Modeling High-resolution Spectroscopic Data](https://arxiv.org/abs/2210.01827)
	:(telluric line fitting?)"transfer learning"—first pretraining models on noise-free precomputed synthetic spectral models, then learning the corrections to line depths and widths from whole-spectrum fitting to an observed spectrum. 
-	[Decoding astronomical spectra using machine learning](https://discovery.ucl.ac.uk/id/eprint/10150994/)
 : automatic line fitting through machine learning
-	[Examining quasar absorption-line analysis methods: the tension between simulations and observational assumptions key to modelling clouds](https://arxiv.org/abs/2202.12228) 
  : Voigt profile is not really accurate?



# ?
- What should be the final result form? software program? paper?
- The gap between the synthetic / real data may not convincing
- Is leveraging machine learning the right approach? Machine learning is not 100% accurate.
- How to define 'accuracy' ? The p
