
# Import libraries
from textattack.augmentation import EasyDataAugmenter, CheckListAugmenter

import config as cfg
from nlp_ta_augmenter import augment_data, create_custom_augmenter
from nlp_ta_handler import fetch_and_clean_data, save_pkl_file



# Load data
cleaned_data, target = fetch_and_clean_data(cfg.categories)

# EDA Augmentation
eda_augmenter = EasyDataAugmenter(
    pct_words_to_swap=cfg.pct_words2swap_value, 
    transformations_per_example=cfg.transformations_example_value
    )

augmented_data_eda, augmented_label_eda = augment_data(cleaned_data, target, eda_augmenter)

# Save
save_pkl_file('./data/augmented_data_eda.pkl', augmented_data_eda)
save_pkl_file('./data/augmented_label_eda.pkl', augmented_label_eda)


# CKL Augmentation
ckl_augmenter = CheckListAugmenter(
    pct_words_to_swap=cfg.pct_words2swap_value, 
    transformations_per_example=cfg.transformations_example_value
    )

augmented_data_ckl, augmented_label_ckl = augment_data(cleaned_data, target, ckl_augmenter)

# Save
save_pkl_file('./data/augmented_data_ckl.pkl', augmented_data_ckl)
save_pkl_file('./data/augmented_label_ckl.pkl', augmented_label_ckl)

# Customized Augmentation
ca_augmenter= create_custom_augmenter(
    pct_words_to_swap=cfg.pct_words2swap_value, 
    transformations_per_example=cfg.transformations_example_value
    )

augmented_data_ca, augmented_label_ca = augment_data(cleaned_data, target, ca_augmenter)

# Save
save_pkl_file('./data/augmented_data_ca.pkl', augmented_data_ca)
save_pkl_file('./data/augmented_label_ca.pkl', augmented_label_ca)






















