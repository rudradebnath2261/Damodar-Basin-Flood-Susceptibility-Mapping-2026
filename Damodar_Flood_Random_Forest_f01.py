"""Damodar Basin flood susceptibility mapping using Random Forest.
Adapted from the supplied landslide Random Forest workflow.
Requires: numpy, pandas, geopandas, rasterio, scikit-learn, joblib.
"""
import os
import re
from contextlib import ExitStack

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split


# 1. Set up input files and output folder

folder = r"C:\ADAMAS\rdx\Flood_GEOAI_Damodar Basin flood susceptibility map\Flood_GEOAI"
points_file = os.path.join(folder, "DB_FloodData", "DB_FloodData.shp")
output_folder = os.path.join(folder, "output")
os.makedirs(output_folder, exist_ok=True)

# These are the raster layers used to predict flooding.
raster_files = {
    "Dist_Builtup": os.path.join(folder, "D_From_Builtup", "Distance_from_builtup.tif"),
    "Dist_River": os.path.join(folder, "D_from_river", "distance.tif"),
    "Drainage_Density": os.path.join(folder, "Drainage_Density", "Density.tif"),
    "Elevation": os.path.join(folder, "Elevation", "DB_DEM.tif"),
    "LULC": os.path.join(folder, "LULC", "Dhanbad_LULC_5_Class.tif"),
    "Population": os.path.join(folder, "Population", "Pop.tif"),
    "Rainfall": os.path.join(folder, "Slope", "Precipitation", "Rainfall.tif"),
    "Slope": os.path.join(folder, "Slope", "Clip_Slope1.tif"),
    "Water_Discharge": os.path.join(folder, "Water_Discharge", "Water_Discharge1.tif"),
}
predictors = list(raster_files)

# A probability of 0.50 or higher means the pixel is flood susceptible.
THRESHOLD = 0.50
TILE_SIZE = 512
RANDOM_STATE = 42
NODATA_FLOAT = -9999.0
NODATA_BYTE = 255


# 2. Check the input files and flood sample points

for label, path in [("Flood points", points_file), *raster_files.items()]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing input '{label}': {path}")

gdf = gpd.read_file(points_file)
if gdf.empty:
    raise ValueError("Flood point shapefile is empty.")
if gdf.crs is None:
    raise ValueError("Flood point shapefile has no CRS. Define its correct CRS in GIS first.")
if not gdf.geometry.geom_type.isin(["Point"]).all():
    raise ValueError("Flood sample shapefile must contain Point features only.")

# Find the flood occurrence field, even if the shapefile has shortened its name.
def normalized(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())

candidates = ["Flood_Occu", "Flood_Occ", "FloodOccurrence", "Flood_Occurrence",
              "Flood Occu", "FloodOccu", "Flood", "Flooded"]
matched = [col for col in gdf.columns if normalized(col) in
           {normalized(x) for x in candidates}]
if not matched:
    matched = [col for col in gdf.columns if normalized(col).startswith("floodocc")]
if len(matched) != 1:
    raise ValueError(
        "Could not uniquely identify Flood Occurrence label. "
        f"Fields present: {list(gdf.columns)}. "
        "Edit 'candidates' to include your exact flood occurrence field."
    )
target = matched[0]
print("Flood target field:", target)


# 3. Align all rasters with the elevation raster

# Use bilinear resampling for continuous rasters and nearest neighbour for LULC.
# Use the same grid for both sample extraction and the final maps.
with ExitStack() as stack:
    ref_src = stack.enter_context(rasterio.open(raster_files["Elevation"]))
    if ref_src.crs is None:
        raise ValueError("Elevation raster is missing CRS.")
    reference_crs = ref_src.crs
    ref_transform = ref_src.transform
    height, width = ref_src.height, ref_src.width
    profile = ref_src.profile.copy()

    vrt_layers = {}
    for name, path in raster_files.items():
        src = stack.enter_context(rasterio.open(path))
        if src.crs is None:
            raise ValueError(f"Raster '{name}' is missing CRS: {path}")
        vrt = stack.enter_context(WarpedVRT(
            src,
            crs=reference_crs,
            transform=ref_transform,
            width=width,
            height=height,
            resampling=(Resampling.nearest if name == "LULC" else Resampling.bilinear),
            dtype="float32",
            nodata=NODATA_FLOAT,
        ))
        vrt_layers[name] = vrt


    # 4. Get raster values at each flood sample point
 
    sample_gdf = gdf.to_crs(reference_crs)
    coords = [(geom.x, geom.y) for geom in sample_gdf.geometry]
    samples = {}
    for name, vrt in vrt_layers.items():
        values = np.array([np.nan if np.ma.is_masked(v[0]) else float(v[0])
                           for v in vrt.sample(coords, masked=True)], dtype="float32")
        samples[name] = values

    # Make a table of the extracted values.
    data = pd.DataFrame(samples)
    data["Flood_Label"] = pd.to_numeric(sample_gdf[target], errors="coerce").to_numpy()
    data = data.replace([np.inf, -np.inf, NODATA_FLOAT], np.nan).dropna()
    data = data[data["Flood_Label"].isin([0, 1])].copy()
    data["Flood_Label"] = data["Flood_Label"].astype(int)

    print("Input points:", len(gdf))
    print("Usable sample points:", len(data))
    print("Class counts:\n", data["Flood_Label"].value_counts())
    if data["Flood_Label"].nunique() != 2:
        raise ValueError("Valid training data must contain both flood (1) and non-flood (0).")
    if data["Flood_Label"].value_counts().min() < 2:
        raise ValueError("Each class needs at least two valid points for stratified splitting.")

    data.to_csv(os.path.join(output_folder, "flood_training_samples.csv"), index=False)
    X = data[predictors]
    y = data["Flood_Label"]

    # 5. Divide samples into training (80%) and testing (20%) data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # --------------------------------------------------
    # 6. Train the Random Forest model

    # Train with 200 trees and account for class imbalance.
    model = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump({"model": model, "predictors": predictors, "threshold": THRESHOLD},
                os.path.join(output_folder, "flood_random_forest_model.joblib"))


    # 7. Check model performance using test samples

    pred = model.predict(X_test)
    prob_test = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    auc = roc_auc_score(y_test, prob_test) if y_test.nunique() == 2 else float("nan")
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC-AUC: {auc:.4f}")
    print("Classification report:\n", report)
    print("Confusion matrix [rows=actual, columns=predicted; 0, 1]:\n", cm)

    with open(os.path.join(output_folder, "model_accuracy_report.txt"), "w") as f:
        f.write(f"Flood target field: {target}\n")
        f.write(f"Usable training/test points: {len(data)}\n")
        f.write(f"Train count: {len(X_train)}; Test count: {len(X_test)}\n")
        f.write(f"Test accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)\n")
        f.write(f"ROC-AUC: {auc:.6f}\n\n")
        f.write("Classification report:\n" + report + "\n")
        f.write("Confusion matrix (actual rows; predicted columns; 0,1):\n")
        f.write(str(cm) + "\n")
        f.write("Note: Random point split can overestimate spatial generalization.\n")

    pd.DataFrame(cm, index=["Actual_0", "Actual_1"],
                 columns=["Predicted_0", "Predicted_1"]).to_csv(
                     os.path.join(output_folder, "confusion_matrix.csv"))
    pd.DataFrame({"Predictor": predictors,
                  "Importance": model.feature_importances_}).sort_values(
                      "Importance", ascending=False).to_csv(
                          os.path.join(output_folder, "feature_importance.csv"), index=False)


    # 8. Create the probability and binary flood maps

    # Process the rasters in smaller tiles to save memory.
    base_profile = profile.copy()
    base_profile.update(driver="GTiff", count=1, compress="lzw")
    base_profile.pop("photometric", None)
    float_profile = base_profile.copy()
    float_profile.update(dtype="float32", nodata=NODATA_FLOAT)
    byte_profile = base_profile.copy()
    byte_profile.update(dtype="uint8", nodata=NODATA_BYTE)

    # Choose where to save the two final maps.
    prob_path = os.path.join(output_folder, "flood_probability.tif")
    binary_path = os.path.join(output_folder, "flood_binary.tif")
    with rasterio.open(prob_path, "w", **float_profile) as prob_dst, \
         rasterio.open(binary_path, "w", **byte_profile) as binary_dst:
        for top in range(0, height, TILE_SIZE):
            for left in range(0, width, TILE_SIZE):
                window = Window(left, top,
                                min(TILE_SIZE, width-left),
                                min(TILE_SIZE, height-top))
                arrays = [vrt_layers[name].read(1, window=window, masked=True)
                          for name in predictors]
                stack_data = np.ma.stack(arrays, axis=-1)
                invalid = np.ma.getmaskarray(stack_data).any(axis=-1)
                values = stack_data.filled(np.nan)
                valid = (~invalid) & np.all(np.isfinite(values), axis=-1) & \
                        np.all(values != NODATA_FLOAT, axis=-1)
                prob_out = np.full(valid.shape, NODATA_FLOAT, dtype="float32")
                binary_out = np.full(valid.shape, NODATA_BYTE, dtype="uint8")
                if np.any(valid):
                    pixel_df = pd.DataFrame(values[valid], columns=predictors)
                    p = model.predict_proba(pixel_df)[:, list(model.classes_).index(1)]
                    prob_out[valid] = p.astype("float32")
                    binary_out[valid] = (p >= THRESHOLD).astype("uint8")
                prob_dst.write(prob_out, 1, window=window)
                binary_dst.write(binary_out, 1, window=window)

print("\nFlood susceptibility mapping completed!")
print("Results saved in:", output_folder)
print("Probability map:", prob_path)
print("Binary flood map:", binary_path)
