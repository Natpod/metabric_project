# METABRIC Breast Cancer Clinical Data Analysis Project
1) Background
Breast cancer is a disease in which abnormal cells in the breast grow uncontrollably and form 
tumors. If not detected and treated early, these tumors can spread to other parts of the body and become life-threatening. It is the most common cancer among women worldwide, accounting for 
approximately 25% of all cancer cases. In 2023, it affected around 2.2 million people and ranked as the second leading cause of cancer-related deaths in women.

Early diagnosis and accurate characterization of tumor types significantly improve survival rates. In this context, data-driven approaches and machine learning techniques have the potential to enhance diagnosis, optimize treatment selection, and improve patient monitoring. These advances can ultimately reduce the risk of recurrence and improve long-term outcomes for cancer patients and survivors. This project aims to:

## Repository Structure and Workflow

The `data`directory contains the .csv file that comes from The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) database is a Canada-UK Project which contains targeted sequencing data of 1,980 primary breast cancer samples (Pereira et al., 2016). 

Schema:


Sequence of scripts
QC > EDA > Decision Making on Models > Training > Inference

```
conda create -n metabric python=3.10 --y
conda activate metabric
pip intall -r requirements.txt
python src/main.py --no-show-plots

```


# OBJECTIVES

* Perform Exploratory Analysis on Data to derive potential machine learning models that could be applied in a clinical context to improve 

- diagnosis `'cancer_type','cancer_type_detailed','tumor_other_histologic_subtype','oncotree_code'`
- treatment: target variables: `'chemotherapy','hormone_therapy','radio_therapy''type_of_breast_surgery`

- recovery and follow-up `'death_from_cancer','overall_survival_months','overall_survival'`

of cancer patients


HOW FEATURES ARE SEGMENTATED HETEROGENEOUS BY 'age_at_diagnosis','cohort','ethnicity'

* Implement machine learning models

* Explain the objective of each model and how it could be implemented in a real-world clinical setting

