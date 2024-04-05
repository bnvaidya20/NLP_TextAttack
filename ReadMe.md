# Insights of NLP Text Augmentation with TextAttack

## Overview

This repository delves into the exploration of data augmentation in Natural Language Processing (NLP) using the versatile Python framework, TextAttack. The focal point of this exploration is to augment textual data, enriching the dataset to improve the robustness and performance of NLP models. Text augmentation is a pivotal technique in NLP, aimed at artificially expanding the training dataset by applying a series of transformations to the text, thereby simulating linguistic variations. This process can lead to models that are more resilient and perform better on diverse or unseen data.

TextAttack offers a rich set of tools for conducting text augmentation, including but not limited to synonym replacement, word swapping, and character manipulation. This project leverages these capabilities to investigate the impact of different augmentation strategies on model accuracy. 

Here are some key points to consider:

### Purpose and Use Cases of TextAttack

- **Adversarial Testing**: TextAttack can be used to test the robustness of NLP models by generating adversarial examples that challenge the models in controlled ways. This is similar to stress-testing models to identify and fix vulnerabilities, ultimately making them more reliable.
- **Data Augmentation**: It offers techniques to augment text data, which can help in overcoming challenges like dataset scarcity and model overfitting, thereby improving the diversity and size of training data.
- **Research and Development**: TextAttack facilitates research in NLP by providing a unified platform for adversarial attacks, augmentations, and training, which can lead to advancements in the field.

### Ethical Considerations and Misuse of TextAttack
- Like many tools in machine learning and cybersecurity, the potential for misuse exists depending on the intent of the user. TextAttack could be misused to craft and deploy adversarial examples against NLP systems in unethical ways, potentially leading to misinformation or exploitation of system vulnerabilities.
- It's crucial for users to adhere to ethical guidelines and best practices, ensuring that TextAttack is used to enhance and secure systems rather than exploit them.

# Project Components

The exploration is structured around three sub-projects:

## 1. **Leveraging TextAttack for Augmenting NLP Datasets **

This script (`nlp_textattack.py`) demonstrates the application of TextAttack, a library designed for adversarial attacks, augmentations, and training in NLP, to augment text data using different strategies.

### Data Preprocessing
- **Data Acquisition:** Utilizes `fetch_20newsgroups`, a popular text dataset for NLP tasks, selecting specific categories like computer hardware, graphics, autos, and cryptography.
- **Cleaning Pipeline:** Employs BeautifulSoup and unidecode to clean the text data, removing HTML tags and normalizing unicode characters, respectively.

### Augmentation Strategies
- **EasyDataAugmenter (EDA):** A basic augmentation strategy provided by TextAttack, which includes synonym replacement, random insertion, random swap, and random deletion.
- **CheckListAugmenter (CKL):** An advanced strategy that applies a checklist of linguistic transformations to simulate various textual perturbations.
- **Custom Augmenter:** A user-defined augmenter that combines `WordSwapQWERTY` and `WordSwapRandomCharacterDeletion` transformations, controlled by parameters such as `pct_words_to_swap` and `transformations_per_example`.

## 2. **Evaluating NLP Models with TextAttack Augmented Data **

This script (`nlp_textattack_evaluate.py`) is structured to perform a comprehensive evaluation of various machine learning models using NLP datasets augmented through TextAttack. 

### Data Handling and Feature Extraction
- **Data Loading:** Pickle files are used to load augmented datasets generated through different TextAttack augmentations (EasyDataAugmenter, CheckListAugmenter, and Custom Augmentation).
- **Feature Extraction:** Utilizes `CountVectorizer` to transform textual data into a numerical format suitable for machine learning models, effectively capturing the vocabulary and occurrence of words within the datasets.

### Model Evaluation Framework
- **Classifiers:** The script evaluates a variety of models, including Ridge Classifier, Multinomial Naive Bayes, Random Forest, and XGBoost, showcasing a broad spectrum of approaches from linear models to ensemble and boosting methods.
- **Cross-Validation:** Cross-validation is used to ensure that each fold is a good representative of the whole, providing a robust evaluation of the model's performance across different subsets of the data.
- **Performance Metrics:** The script computes accuracy, classification reports, and confusion matrices, offering a comprehensive overview of the models' effectiveness in classifying the text data accurately.

### Visualization and Reporting
- **Confusion Matrix Visualization:** Provides a visual representation of the model performance, highlighting the true positives, false positives, true negatives, and false negatives, normalized to show percentages, which is crucial for understanding the model's behavior across different classes.
- **Detailed Analysis:** Through classification reports and confusion matrices, the script offers an in-depth look at precision, recall, f1-score, and support for each class, allowing for a nuanced analysis of model performance, including areas of strength and weakness.

## 3. **Exploration of NLP Text Augmentation using TextAttack **

This script (`nlp_textattack_explore.py`) leverages these capabilities to investigate the impact of different augmentation parameters on model accuracy.

### Data Preparation and Augmentation 
- **Data Sourcing:** Utilizes the `fetch_20newsgroups` dataset, focusing on specific categories such as computer hardware, graphics, autos, and cryptography.
- **Cleaning Methods:** Employs BeautifulSoup and unidecode for HTML tag removal and character encoding normalization, ensuring data uniformity.
- **Augmentation Strategy:** Implements a custom-built augmenter setup, to generate augmented datasets. Parameters such as `pct_words_to_swap` and `transformations_per_example` are systematically varied to assess their influence on the augmentation's effectiveness.

### Feature Extraction and Model Training
- Extracts features from the augmented text data using `CountVectorizer`.
- Splits the data into training and testing sets to ensure a fair evaluation of the models.
- Trains RandomForestClassifier on the augmented data to observe the impact of text augmentation on model performance.

### Evaluation and Visualization
- Evaluates model performance using accuracy metric on a test set.
- Generates a heatmap and a 3D surface plot to visually represent how varying augmentation parameters (e.g., `pct_words_to_swap` and `transformations_per_example`) affect model accuracy. This visual representation provides intuitive insights into the augmentation process's effectiveness.


## References

- [TextAttack Documentation](https://textattack.readthedocs.io/en/latest/index.html)
- [NLP Augmentation Hands-On Part-1](https://akgeni.medium.com/nlp-augmentation-hands-on-cda88aa5d837)
- [NLP Augmentation Hands-On Part-2](https://akgeni.medium.com/nlp-augmentation-hands-on-77bfd9fff5e2)
