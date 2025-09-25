import os
import tempfile
import pandas as pd
import xarray as xr
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- Helper: Convert Argo JULD to datetime ---
def julian_to_datetime(juld_days):
    """Convert Argo 'JULD' (days since 1950-01-01) into ISO datetime string."""
    if pd.isna(juld_days):
        return None
    try:
        ref_date = datetime(1950, 1, 1)
        return (ref_date + timedelta(days=float(juld_days))).isoformat()
    except Exception:
        return None

# --- Parse Argo NetCDF file ---
def parse_argo_file(file_path):
    try:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            
            # 1. Extract platform numbers (array, not scalar)
            platform_numbers = [str(p).strip() for p in ds["PLATFORM_NUMBER"].values]

            # 2. Cycle-level metadata
            cycle_data_df = pd.DataFrame({
                "cycle_id": ds["CYCLE_NUMBER"].values,
                "juld_days": ds["JULD"].values,
                "latitude": ds["LATITUDE"].values,
                "longitude": ds["LONGITUDE"].values,
                "data_mode": ds["DATA_MODE"].values.astype(str),
                "wmo_float_id": platform_numbers
            })
            cycle_data_df["date_time_utc"] = cycle_data_df["juld_days"].apply(julian_to_datetime)

            # 3. Extract measurement variables safely
            required_vars = ["PRES", "TEMP_ADJUSTED", "PSAL_ADJUSTED"]
            missing_vars = [v for v in required_vars if v not in ds.variables]
            if missing_vars:
                raise ValueError(f"Missing variables in file: {missing_vars}")

            pres = ds["PRES"].values
            temp = ds["TEMP_ADJUSTED"].values
            psal = ds["PSAL_ADJUSTED"].values

            n_prof, n_levels = pres.shape
            records = []

            for i in range(n_prof):
                for j in range(n_levels):
                    # Skip entirely missing levels
                    if np.isnan(pres[i, j]) and np.isnan(temp[i, j]) and np.isnan(psal[i, j]):
                        continue
                    
                    records.append({
                        "wmo_float_id": platform_numbers[i],
                        "cycle_id": int(ds["CYCLE_NUMBER"].values[i]),
                        "date_time_utc": julian_to_datetime(ds["JULD"].values[i]),
                        "latitude": float(ds["LATITUDE"].values[i]),
                        "longitude": float(ds["LONGITUDE"].values[i]),
                        "pressure_dbar": float(pres[i, j]) if not np.isnan(pres[i, j]) else None,
                        "temperature_degC": float(temp[i, j]) if not np.isnan(temp[i, j]) else None,
                        "salinity_psu": float(psal[i, j]) if not np.isnan(psal[i, j]) else None,
                        "data_mode": str(ds["DATA_MODE"].values[i])
                    })

            measurements_df = pd.DataFrame(records)

            return {
                "wmo_float_id": list(set(platform_numbers)),
                "cycle_data": cycle_data_df.to_dict(orient="records"),
                "measurements": measurements_df.to_dict(orient="records")
            }

    except Exception as e:
        logging.error(f"Error during NetCDF parsing: {e}")
        return {"error": str(e)}

# --- Flask Route ---
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
        file.save(tmp_file.name)
        logging.info(f"Processing temporary file: {tmp_file.name}")
        result = parse_argo_file(tmp_file.name)
        os.unlink(tmp_file.name)

    return jsonify(result)

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
