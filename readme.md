Machine Learning Applications in Exoplanet Host Star Recommendation & Determination: Thesis Resources

This repository contains all the necessary supplementary resources for my thesis. The trained ML models are not available here due to space constraints but can be made available upon reasonable request.

Chapter 3: Spectral Classifier Validation Run Resources

    - obs_list_chap3.csv: Observation ID List for the Keck-HIRES dataset
		
    - target_labels_chap3.csv: Labels for all stars included in the Keck-HIRES dataset

Chapter 4: Spectral Classifier & Machine Vision Resources

	- Fetch_spectra.py: Script for programmatic access to the ESO Science Archive Facility and spectral dataset 
	                    downloads
		
	- Datasets
		- Labelled_Dataset.csv: Target Labels, Observation ID and Facility of Observation for all instances 
		                        in the labelled HARPS-FEROS dataset.
		- Unlabelled_Dataset.csv: Target Labels, Observation ID and Facility of Observation for all instances 
		                          in the unlabelled HARPS-FEROS dataset.
  		- ESO_Programmes
                - ESO_Programmes_Labelled_Dataset.csv: List of ESO Programmes in the labelled HARPS-FEROS dataset.
                - ESO_Programmes_Unlabelled_Dataset.csv: List of ESO Programmes in the unlabelled HARPS-FEROS dataset.
    	- PreProcessingPipeline: Collection of scripts necessary to process the raw data and prepare it for input 
		                         into the trained models. Runs through main.py with configurable setting through 
		                         config.ini


Chapter 5: Abundance MOO Engine Resources

	- Pipeline: Implementations of the MOO algorithm for all cases presented in the thesis
 		- Binary_Implementation: Implementations for the classifier selection runs and use-case I
   			- ANNImplementation: Implementation using the Dense Neural Network Classifier
	  		- SVMImplementation: Implementation using the SVM Classifier
	 		- XGBoostImplementation: Implementation using the XGBoost Classifier (Use-Case I)
		- Multilabel_Implementation__inclPlMultiplicity: Implementation for the multilabel run, including the 
	 	                                                 Planet Multiplicity label (Use-Case IIA)
   		- Multilabel_Implementation__exclPlMultiplicity: Implementation for the multilabel run, excluding the 
	 	                                                 Planet Multiplicity label (Use-Case IIB)
													
	- Thresholds for Completeness: Imputation completeness thresholds for every feature within each feature variant in 
	                               all 3 use-cases
		- thresholds_CaseI_implementation: Thresholds for the Binary Implementation
  		- thresholds_CaseIIA_implementation: Thresholds for the Multilabel Implementation, including the Planet 
		                                     Multiplicity label
		- thresholds_CaseIIB_implementation: Thresholds for the Multilabel Implementation, excluding the Planet 
		                                     Multiplicity label
