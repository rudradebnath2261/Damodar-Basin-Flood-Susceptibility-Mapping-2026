# Damodar-Basin-Flood-Susceptibility-Mapping-2026
Flood susceptibility mapping of the Damodar Basin using GIS, remote sensing, and an optimized Random Forest machine learning model. Integrates environmental and hydrological factors, hyperparameter tuning, and spatial analysis to predict flood-prone zones, assess model accuracy, and support disaster risk management.
# Flood Susceptibility Mapping of the Damodar Basin Using Random Forest Machine Learning

## 1. Project Overview

This project focuses on flood susceptibility assessment in the **Damodar Basin, India**, using Geographic Information Systems (GIS), remote sensing, and an optimized Random Forest machine learning algorithm. The study aims to identify flood-prone areas by examining the relationship between historical flood occurrence locations and environmental, topographical, hydrological, and anthropogenic factors.

A GIS-based machine learning framework was developed using Python to integrate multiple spatial datasets, train a Random Forest classification model, evaluate its predictive performance, and generate spatially explicit flood susceptibility maps.

The project supports flood hazard assessment, disaster risk reduction, regional planning, and improved understanding of the environmental factors influencing flood susceptibility.

## 2. Research Objectives

- Identify and map flood-susceptible locations within the Damodar Basin.
- Integrate topographical, hydrological, environmental, and anthropogenic variables using GIS and remote sensing.
- Develop a Random Forest machine learning model using historical flood occurrence and non-occurrence samples.
- Optimize model performance through hyperparameter tuning and cross-validation.
- Evaluate classification performance using accuracy, precision, recall, F1-score, balanced accuracy, ROC-AUC, and a confusion matrix.
- Generate continuous flood susceptibility probability maps and binary flood classification maps.
- Examine the relative contribution of different environmental predictors to flood susceptibility.

## 3. Study Area

The study focuses on the Damodar Basin in eastern India, a river basin extending through parts of Jharkhand and West Bengal. The region experiences seasonal monsoon rainfall and is influenced by variations in elevation, slope, drainage characteristics, land use, river proximity, and hydrological conditions.

These factors influence the spatial distribution of areas potentially susceptible to flooding.

The actual mapping extent is determined by the input elevation raster and the availability of the remaining predictor datasets.

## 4. Datasets and Flood Conditioning Factors

Nine spatial conditioning factors were integrated into the machine learning framework.

| Sl. No. | Conditioning factor | Dataset |
|---|---|---|
| 1 | Distance from built-up areas | Distance_from_builtup.tif |
| 2 | Distance from rivers | distance.tif |
| 3 | Drainage density | Density.tif |
| 4 | Elevation | DB_DEM.tif |
| 5 | Land Use and Land Cover (LULC) | Dhanbad_LULC_5_Class.tif |
| 6 | Population distribution | Pop.tif |
| 7 | Rainfall | Rainfall.tif |
| 8 | Terrain slope | Clip_Slope1.tif |
| 9 | Water discharge | Water_Discharge1.tif |

Historical flood occurrence and non-occurrence locations were obtained from the `DB_FloodData.shp` point dataset and used as reference observations for supervised classification.

The target classes are defined as:

- **Class 1:** Flood occurrence
- **Class 0:** Non-flood occurrence

The source organizations, acquisition dates, spatial resolution, and temporal coverage of the input datasets should be documented separately when their original metadata are available.

## 5. Methodology

### Step 1: Data Collection and Preprocessing

The study integrates raster-based geographical and environmental datasets with historical flood occurrence information.

The elevation raster was selected as the reference grid for spatial alignment. All predictor rasters were reprojected and resampled to match the coordinate reference system, pixel size, geographical extent, and grid alignment of the elevation dataset.

Bilinear resampling was applied to continuous environmental variables, while nearest-neighbor resampling was used for categorical LULC data to preserve land-cover class values.

Invalid observations, missing raster values, and NoData pixels were excluded from model training and prediction.

### Step 2: Training Sample Preparation

The historical flood occurrence point shapefile was used to prepare the supervised learning dataset.

Raster values corresponding to the nine conditioning factors were extracted at individual point locations.

Each observation was assigned a binary target representing flood occurrence or non-occurrence.

The resulting dataset contained the conditioning factors as independent variables and historical flood occurrence as the dependent variable.

### Step 3: Training and Testing Data Separation

The dataset was divided into two subsets using stratified random sampling:

- **80%:** Model training
- **20%:** Model testing

A fixed random state of 42 was applied to maintain reproducibility.

Stratification was used to preserve approximately similar flood and non-flood class proportions in both datasets.

### Step 4: Random Forest Model Development

Random Forest is a supervised ensemble machine learning algorithm that combines multiple decision trees to perform classification.

Each decision tree learns relationships between flood occurrence and the selected conditioning factors. The combined predictions are used to estimate flood susceptibility.

The initial Random Forest model used 200 decision trees, balanced class weights, and a fixed random state.

To improve model selection, an optimized Random Forest framework was subsequently developed using RandomizedSearchCV.

### Step 5: Hyperparameter Optimization

Hyperparameter tuning was implemented to identify suitable Random Forest configurations.

The optimization procedure evaluated 30 randomly selected combinations from the following search space:

| Hyperparameter | Values tested |
|---|---|
| Number of trees | 100, 200, 300, 500 |
| Maximum tree depth | None, 5, 10, 15, 20 |
| Minimum samples for splitting | 2, 4, 6, 10 |
| Minimum samples per leaf | 1, 2, 3, 5 |
| Maximum features | sqrt, log2, 0.8 |
| Class weighting | Balanced |

Stratified K-Fold cross-validation was applied within the training dataset, using up to five folds depending on the available number of samples.

The optimization criterion was **balanced accuracy**, which considers classification performance across both flood and non-flood classes.

The best-performing configuration was selected and used to generate flood susceptibility predictions.

### Step 6: Flood Susceptibility Prediction

The optimized Random Forest model estimates the probability of flood occurrence for each valid raster pixel.

The predicted probability is represented as:

\[
P(Flood=1 \mid X)
\]

where \(X\) represents the nine conditioning factors.

The probability values range from 0 to 1, with higher values indicating greater model-estimated flood susceptibility.

A probability threshold of 0.50 was applied to generate the binary classification map:

\[
Flood =
\begin{cases}
1, & P \geq 0.50 \\
0, & P < 0.50
\end{cases}
\]

The final output consists of a continuous flood susceptibility probability raster and a binary flood classification raster.

Raster predictions were processed in spatial tiles to improve memory efficiency during analysis.

## 6. Model Accuracy and Performance Evaluation

The initial Random Forest model was evaluated using an independent 20% random holdout from the supplied sample dataset.

### Initial Model Performance

| Performance metric | Result |
|---|---|
| Overall Accuracy | 71.43% |
| ROC-AUC | 0.5833 |
| Balanced Accuracy | 66.67% |
| Precision – Non-flood | 0.67 |
| Recall – Non-flood | 1.00 |
| F1-score – Non-flood | 0.80 |
| Precision – Flood | 1.00 |
| Recall – Flood | 0.33 |
| F1-score – Flood | 0.50 |


### Confusion Matrix

| Actual / Predicted | Non-flood (0) | Flood (1) |
|---|---|---|
| Non-flood (0) | 4 | 0 |
| Flood (1) | 2 | 1 |

The initial model correctly classified five out of seven testing observations, producing an overall accuracy of **71.43%**.

The model successfully classified all four non-flood observations but identified only one of the three actual flood observations. The flood-class recall of 33.33% indicates that the model missed a substantial proportion of flood observations in the small testing dataset.

The ROC-AUC value of 0.5833 further suggests limited discrimination between flood and non-flood classes.

These results highlight the need for additional training samples, improved spatial sampling, predictor evaluation, and model optimization.

### Improved Random Forest Model

An improved Random Forest framework was developed using hyperparameter optimization, balanced class weighting, and stratified cross-validation.

The optimized model evaluates the following performance measures:

- Overall accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Predictor feature importance

**Optimized model performance:** To be updated using the results generated by the improved Random Forest script.

The initial accuracy of 71.43% should not be interpreted as the final performance of the optimized model.

## 7. Model Outputs

The Python workflow generates the following outputs:

| Output file | Description |
|---|---|
| `flood_probability.tif` | Continuous flood susceptibility probability raster |
| `flood_binary.tif` | Binary flood and non-flood classification raster |
| `flood_random_forest_model.joblib` | Saved optimized Random Forest model |
| `model_accuracy_report.txt` | Model evaluation and performance metrics |
| `confusion_matrix.csv` | Actual versus predicted classification results |
| `feature_importance.csv` | Relative importance of flood conditioning factors |
| `hyperparameter_search_results.csv` | Cross-validation and hyperparameter search results |
| `flood_training_samples.csv` | Training sample attributes and extracted raster values |

The probability raster can be visualized in ArcGIS Pro or QGIS to identify spatial variations in flood susceptibility.

It may also be classified into five susceptibility categories—Very Low, Low, Moderate, High, and Very High—using a separately defined and documented classification scheme.

## 8. Software and Technologies

**Programming Language:** Python

**GIS and Remote Sensing Software:** ArcGIS Pro and QGIS

**Python Libraries:**

- NumPy – Numerical computation and raster array processing.
- Pandas – Data preparation and tabular analysis.
- GeoPandas – Spatial vector data processing.
- Rasterio – Raster reading, alignment, reprojection, and GeoTIFF export.
- Scikit-learn – Random Forest modelling, hyperparameter optimization, cross-validation, and accuracy assessment.
- Joblib – Trained machine learning model serialization.

## 9. Limitations and Future Improvements

The model's reliability depends on the quantity, spatial distribution, temporal relevance, and quality of the historical flood observations and conditioning factors.


Furthermore, randomly separated training and testing observations may be spatially correlated, potentially leading to optimistic estimates of predictive performance.

Future improvements will focus on expanding the historical flood inventory, incorporating independently verified flood and non-flood samples, introducing additional hydrological indicators such as the Topographic Wetness Index (TWI) and Height Above Nearest Drainage (HAND), and conducting spatial cross-validation.

Comparative modelling using alternative machine learning algorithms, including XGBoost and Support Vector Machine (SVM), may also be explored.

## 10. Conclusion

This project demonstrates the application of GIS, remote sensing, and Random Forest machine learning for spatial flood susceptibility assessment in the Damodar Basin.

By integrating nine flood conditioning factors with historical flood occurrence observations, the developed framework produces continuous flood susceptibility probability maps, binary classification maps, model accuracy reports, and feature importance assessments.

The initial Random Forest model achieved **71.43% accuracy**, although its low flood recall and ROC-AUC indicate the need for further refinement.

The implementation of hyperparameter optimization and stratified cross-validation provides a systematic framework for improving model selection and evaluating predictive performance.

With additional field-verified observations, improved environmental data, and spatially independent validation, the approach can support more reliable flood hazard assessment, disaster management, and evidence-based regional planning.

---

**Project Area:** Damodar Basin, India  
**Research Domain:** Geoinformatics, GIS, Remote Sensing, GeoAI, Machine Learning, Flood Susceptibility Mapping  
**Machine Learning Algorithm:** Random Forest Classifier  
**Model Optimization:** RandomizedSearchCV and Stratified K-Fold Cross-Validation  
**Initial Model Accuracy:** 71.43%  
**Optimized Model Accuracy:** Pending evaluation
