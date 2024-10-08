

# Categorizing Religious Online Hate Speech

Programming project by Sofia Carpentieri\
Supervised by Janis Goldzycher\
Departement of Computational Linguistics\
University of Zurich


## Overview

This project contains a pipeline to analyse the dataset collected by [Papasavva et al. (2020)](https://arxiv.org/pdf/2001.07487.pdf) (```analysis.py```), the code for the evaluation of the pre-trained models used in the analysis (```prompting.py```) and a Jupyter-notebook to visualize the data (```plotting.ipynb```). 


## Set-up

To run the analysis, the dataset by [Papasavva et al. (2020)](https://arxiv.org/pdf/2001.07487.pdf) must be downloaded and chunked. Please do the following steps to prepare the analysis:

1. Download and unzip the file named ```pol_0616-1119_labeled.tar.zst``` from [zenodo](https://zenodo.org/records/3606810).
2. Extract the data as described on the website (depending on your system).
3. Make sure the extracted ```ndjson```-file is on a local directory.
4. Create a new directory (e.g., called ```chunks```) to store the new files and change into it.
5. Chunk the dataset using the following bash-command (with the adjusted file-path):
	
	```$ split -l 50000 --numeric-suffixes /local/path/to/pol_062016-112019_labeled.ndjson data_ --additional-suffix=.json```
6. Wait a while for the dataset to be split into 75 files. Then you should be good to go. For running the analysis, you will only need the path to the directory containing all these files.


## Classifier Evaluation

The code for evaluating the classisiers used can be found in the file ```prompting.py```. The code can be run as it is and be adjusted by uncommenting the functions and the various approaches e.g., different hypotheses. To run the code, you can use the bash-command ```$ python3 prompting.py```. Each evaluating function will additionally print the accuracy and macro F1-score.

* ```predict_hatespeech()``` Evaluate the model used for predicting hate speech
* ```hatespeech_type_classification()``` Evaluate the model used for predicting the type of hate speech using an NLI-approach
* ```count_targets()``` Evaluate the model used for predicting the type of hate speech using a zero-shot-classification-approach
* ```count_religion()``` Evaluate the model used for predicting the target religion using a zero-shot-classification-approach
* ```count_religion_hypothesis()``` Evaluate the model used for predicting the target religion using an NLI-approach
* ```random_baseline()``` Return the values of the random baseline for the target religion prediction
* ```majority_baseline()``` Return accuracy and macro-F1 of the majority baseline (i.e., Judaism) for the target religion prediction



## Dataset Analysis

To run the analysis, use the bash-command ```$ python3 analysis.py``` and input the path to the directory containing all chunked files as created in the set-up. The analysis will run by its own and does not require further input. The pipeline runs over every ```json```-file in the directory, annotates the data and metadata and stores everything in separate files. The pipeline, i.e. the code in the ```main()``` function, uses three classes: 

* ```Preprocessor``` 
    * ```manual_replacement()``` Escape HTML-elements and handle board mentions
    * ```pos_tag()``` Annotate part-of-speech tags, dependencies and linguistic information using spaCy
    * ```preprocess_post()``` Combine the functions mentioned above
* ```DataAnnotator```
    * ```extract_all()``` Extract linguistic information using the ```Preprocessor```
    * ```predict_hatespeech()``` Predict whether a post is hate speech or not
    * ```count_targets()``` Predict wether a post is hate speech against a religion
    * ```count_religion()``` Predict the target religion of a hate speech post
    * ```annotate_data()``` Combine all functions mentioned above and writing the output into separate files
* ```MetaAnalyser```
    * ```get_statistics()``` Get the statistics of the metadata-file
    * ```print_statistics()``` Print statistics of the metadata-file in a readable format



## Visualization of (Meta) Data

The Jupyter-notebook, ```plotting.ipynb```, contains the code to visualize the meta data and data. It prints raw numbers, percentages and creates plots for statistics on countries of origin of the posts, hate speech, religious hate speech and the specific religions targeted. The user only needs to input two filepaths; the path to the file ```meta_overall.tsv``` and to the directory ```extracted```.

## Final directory structure

Once the analysis is finished, your output files are stored in the same directory as the chunked files. As there are many new files created, this section shows how the final directory structure should look like. The top-most directory, ```chunks/```contains all chunked files from the set-up.

```
chunks/
│   data_00.json
│   ...
│   data_74.json
│   meta_data_00.json
│   ...
│   meta_data_74.json
│   meta_overall.tsv
│
└─── extracted/
│   │   extracted_data_00.json
│   │   ...
│   │   extracted_data_74.json
│   
└─── hatespeech/
    │   hatespeech_data_00.txt
    │   ...
    │   hatespeech_data_74.txt
    │   religion_specific_hatespeech_data_00.txt
    │	...
    │   religion_specific_hatespeech_data_74.txt
    │   religious_hatespeech_data_00.txt
    │	...
    │   religious_hatespeech_data_74.txt

```


