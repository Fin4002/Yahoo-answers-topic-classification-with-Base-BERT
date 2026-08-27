CSE440: Natural Language Processing II  
Lab Project

# **1\.  Project Overview**

This is an open-ended, research-oriented lab project. You will independently identify a real-world multi-class text classification problem, source an appropriate dataset, and build a complete text classification pipeline. The goal is to explore how different preprocessing techniques, text representations, and model architectures affect classification performance. You will analyze the dataset, evaluate multiple approaches, and communicate your findings through code, a formal research report, and a recorded presentation.

# **2\.  Dataset & Topic Selection**

You have complete freedom to choose your topic and dataset, subject to the constraints below.

Your chosen dataset must be approved by the lab instructors before you begin experiments.

## **2.1  Dataset Constraints**

* Type: Textual

* Language: English 

* Minimum number of classes: 4 (i.e., at least 4 distinct target labels)

* Minimum dataset size: 10,000 rows/samples (across all classes combined)

* The dataset must be split (or splittable) into training and test sets

* The class distribution must be documented; severe imbalance should be noted and addressed

## **2.2  Topic Constraints**

* The task must be a multi-class text classification problem

* Choose something meaningful and non-trivial; avoid common topics such as sentiment classification, movie recommendation, etc

* If the topic has existing implementations or reported results, then you  should explore ways to improve performance through better preprocessing, feature representations, model architectures, or training strategies


  

## **2.3  Approval Process**

Group and Topic formation must be completed through the provided Lab Project Formation Google Sheet within the specified deadline.

* Submit two proposed topics, each with a one-line description

* Provide the dataset links for both proposed topics

* The lab instructors will approve one topic and its dataset or provide one additional day to update the topics if none of the proposals are suitable

* After approval, you should begin working on the project

# **3\.  Project Guidelines** 

Your project should follow a complete machine learning workflow for multi-class text classification. At a minimum, it should include the following components:

## **3.1  Problem Definition**

* Clearly define the classification task and class label

* Justify the real-world relevance of the problem

## **3.2  Dataset Collection and Description**

* Obtain a suitable multi-class text classification dataset

* Describe the dataset, source, class distribution, and key characteristics

## **3.3  Dataset Collection and Description**

* Analyze necessary data statistics

* Identify potential issues such as class imbalance, duplicates, or missing values

* Present relevant visualizations

## **3.4  Data Preprocessing**

* Clean and preprocess the text data

* Document and justify all preprocessing decisions

* Compare different preprocessing strategies where appropriate

## **3.5  Train/Validation/Test Split**

* Split the dataset into training, validation, and test sets

* Ensure that the split strategy is appropriate and reproducible

## **3.6  Text Representation**

* Implement and evaluate multiple text representation techniques

* You must use at least two of the following approaches: TF-IDF, Word2Vec, or GloVe

## **3.7  Model Development**

* Train all three of the following ML models:

  * Random Forest  
  * Logistic Regression  
  * Naive Bayes

* Train all seven of the following NN models:

  * SimpleRNN

  * GRU

  * LSTM

  * Bidirectional SimpleRNN

  * Bidirectional GRU

  * Bidirectional LSTM

  * BERT Base

* You may train any other advanced models if time permits

## **3.8  Hyperparameter Tuning**

* For every model, you must perform manual hyperparameter tuning guided by validation set performance

* Each model must be trained for at least 3 different hyperparameter configurations

* Record all tuning runs in a dedicated table in your code and report

## **3.9  Model Evaluation**

* Evaluate every model (using its best hyperparameter configuration) on the test set. Report all of the following:

  * Accuracy

  * F1-score 

  * Confusion Matrix

  * Full Classification Report

* At the end of your experiments, you must:

  * Illustrate and compare the performances of the models through tables and figures

  * Identify and highlight the best-performing and worst-performing model

  * Discuss why certain configurations outperformed others; ground your analysis in your EDA findings and the theoretical properties of the methods

# **4\.  Bonus Marks  (+2)**

Bonus marks (+2) will be awarded for work that goes clearly beyond the project requirements. Examples include:

* Highly organized codebase, report, and presentation

* Improving upon existing benchmark results for the chosen topic and dataset

* Implementing an ensemble of the best-performing models and demonstrating improvement

* Conducting ablation studies to analyze the contribution of individual components

* Organizing project code and resources in a GitHub repository (all team members)

* Deploying the best-performing model on a hosting platform (such as Vercel) to allow users to test predictions

Only one team will receive the \+2 bonus marks from each section. The selection will be made by the lab instructors based on the quality of both the required and additional implementations. The bonus marks will not exceed the total lab marks.

# **5\.  Deliverables**

## **5.1  Jupyter Notebook  (.ipynb)**

* The notebook must be well-organized and clearly structured

* Use appropriate Markdown cells for section headings, explanations, and analysis

* Do not add comments to individual lines; you may add above code blocks

* All plots, tables, and evaluation results should be embedded within the notebook

* All code cells must have visible output logs; cells without output cannot be verified

## **5.2  Project Report  (ACL Style PDF)**

Use the official [ACL LaTeX](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj) or [Word](https://2023.aclweb.org/downloads/acl2023.docx) template. You may refer to this [sample paper](https://aclanthology.org/2025.acl-long.5.pdf). The report must be 7 to 8 pages in length, excluding the references section and appendix. The report should include an informative title with author information. The report will be turned into Turnitin; the threshold of plagiarism and AI is 15%.

Required sections:

* Abstract: Brief summary of your approach, embeddings, models, and key findings

* Introduction: Motivation for the chosen task, overview of the dataset, and research questions

* Related Work: Brief review of prior work on your chosen task and the models you used

* Methodology:

  * Dataset: Description, EDA findings with visualisations (word clouds, class distribution, length stats)

  * Preprocessing: details of applied preprocessing and the justification choices

  * Word Representations: description of implemented representations

  * Model Architectures: architecture details and hyperparameter choices for every model

* Results: consolidated comparison table; confusion matrix heatmaps; discussion of best/worst configurations

* Conclusion: Key takeaways, limitations, and suggested future improvements

* References: All papers, tools, libraries, and datasets cited in ACL citation style

## **5.3 Presentation  (.mp4 uploaded in your Google Drive)**

* Total duration: 8 to 12 minutes

* Each group member must present for 2 to 3 minutes. Each member should introduce his/her name in the presentation.

* Presentations of all members should be merged into one single video

* It is a recorded presentation; camera may be turned off

* Present (through screen share) your code during presentation. Slides are not necessary.

* Presention should also cover dataset & motivation, preprocessing decisions, model architectures, results, and conclusions

# **6\.  Viva**

Following the presentation, each group member will be questioned individually by the instructor.

* All group members must be prepared to answer questions on any part of the project, regardless of task division

* You are expected to explain your implementation confidently, including sections where AI tools assisted

* If you cannot explain a piece of code or a modelling decision, you will be penalised

* Questions may cover both the technical implementation and the theoretical concepts behind the methods

# **7\.  Submission Instructions**

Submit the following through the Google Form provided:

* Report PDF file named: GroupNo\_ID1\_ID2\_ID3\_ID4.pdf

* Jupyter Notebook named: GroupNo\_ID1\_ID2\_ID3\_ID4.ipynb

* Link of Presentation video named: GroupNo\_ID1\_ID2\_ID3\_ID4.mp4 (One single video should be uploaded to your own Google Drive with view access on)

* Link of project files organized in your GitHub (optional, for bonus)

* Link of your project deployed in Vercel or any other platform (optional, for bonus)

Submit the report in Turnitin (deadline same as project files submission deadline):

* Enrollment key: 204ec792ea8b9fdeb9c0c392ea22483c

No late submissions will be allowed under any circumstances. The deadlines are given in the CSE440 Combined Lab Discord Server.

# **8\.  Mark Distribution**

| Component | Marks |
| ----- | :---: |
| **Report** | 3 |
| **Presentation** | 2 |
| **Code** | 2 |
| **Viva** | 4 |
| **Total** | 11 |

