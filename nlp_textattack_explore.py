
# Import libraries
from sklearn.ensemble import RandomForestClassifier

from nlp_ta_handler import fetch_and_clean_data, extract_features, save_pkl_file, load_pkl_file, convert_to_dataframe
from nlp_ta_augmenter import create_custom_augmenter, augment_data, plot_heatmap, plot_3d_surface
from nlp_ta_modeler import split_train_test, evaluate_model

import config as cfg


# Load data
cleaned_data, target = fetch_and_clean_data(cfg.categories)


results = []  

for pct_words in cfg.pct_words_to_swap_values:
    for transformations in cfg.transformations_per_example_values:
        # Perform data augmentation with current parameters
        custom_augmenter = create_custom_augmenter(
            pct_words_to_swap=pct_words, 
            transformations_per_example=transformations
            )
        
        augmented_data, augmented_labels = augment_data(cleaned_data, target, custom_augmenter)
        
        # Extract features
        X_features = extract_features(augmented_data)
        
        # Split data
        X_train, X_test, y_train, y_test = split_train_test(X_features, augmented_labels)

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100)  
        clf.fit(X_train, y_train)
        
        # Evaluate accuracy
        _, accuracy = evaluate_model(X_test, y_test, clf, key="CA_RF")
        
        # Store results
        results.append({
            'pct_words_to_swap': pct_words,
            'transformations_per_example': transformations,
            'accuracy': accuracy
        })

save_pkl_file('./data/results.pkl', results)

results= load_pkl_file('./data/results.pkl')

# Print results
for result in results:
    print(f"pct_words_to_swap: {result['pct_words_to_swap']}, transformations_per_example: {result['transformations_per_example']}, Accuracy: {result['accuracy'] * 100:.2f}%")
    

df_aug = convert_to_dataframe(results)

plot_heatmap(df_aug)

plot_3d_surface(df_aug)






