# CS-GO-Round-Winner-Prediction
An end-to-end deep learning pipeline built to predict Counter-Strike: Global Offensive round outcomes (CT vs. T wins) based on mid-game snapshot data.

This project processes a dataset of over 122,000 match records. The data pipeline utilizes pandas and scikit-learn for rigorous data hygiene, handling missing values, applying one-hot encoding to map categories, and standardizing 103 distinct features without data leakage. The core predictive model is a custom multi-layer perceptron (MLP) neural network built with TensorFlow and Keras. Optimized using the Adam algorithm and Binary Crossentropy, the network achieves an 83% overall accuracy on unseen test data, evaluated through detailed classification reports and confusion matrices.
