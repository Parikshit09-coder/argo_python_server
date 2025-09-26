import os
import tempfile
import json
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- Helper: Convert Argo JULD to datetime --- #
def julian_to_datetime(juld_days):
    """Convert Argo 'JULD' (days since 1950-01-01) into ISO datetime string."""
    if pd.isna(juld_days):
        return None
    try:
        ref_date = datetime(1950, 1, 1)
        return (ref_date + timedelta(days=float(juld_days))).isoformat()
    except Exception:
        return None

# --- Parser Function --- #
def parse_argo_file(file_path):
    try:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            # Decode platform numbers
            platform_ids = [
                bytes(p).decode("utf-8").strip() if isinstance(p, (bytes, np.bytes_)) else str(p).strip()
                for p in ds["PLATFORM_NUMBER"].values
            ]
            n_prof = len(platform_ids)

            # Adjusted vs raw data
            pres_adj = ds["PRES_ADJUSTED"].values if "PRES_ADJUSTED" in ds else None
            temp_adj = ds["TEMP_ADJUSTED"].values if "TEMP_ADJUSTED" in ds else None
            psal_adj = ds["PSAL_ADJUSTED"].values if "PSAL_ADJUSTED" in ds else None

            pres_raw = ds["PRES"].values
            temp_raw = ds["TEMP"].values
            psal_raw = ds["PSAL"].values

            data_modes = ds["DATA_MODE"].values

            # Ensure dimensions consistent
            if pres_raw.shape[0] != n_prof:
                pres_raw, temp_raw, psal_raw = pres_raw.T, temp_raw.T, psal_raw.T
                if pres_adj is not None: pres_adj = pres_adj.T
                if temp_adj is not None: temp_adj = temp_adj.T
                if psal_adj is not None: psal_adj = psal_adj.T

            cycle_data, full_arrays = [], []
            for i in range(n_prof):
                profile_id = f"{platform_ids[i]}_{int(ds['CYCLE_NUMBER'].values[i])}"
                current_data_mode = bytes(data_modes[i]).decode("utf-8").strip()

                if current_data_mode in ['D', 'A'] and pres_adj is not None:
                    pres_profile = pres_adj[i]
                    temp_profile = temp_adj[i]
                    psal_profile = psal_adj[i]
                else:
                    pres_profile = pres_raw[i]
                    temp_profile = temp_raw[i]
                    psal_profile = psal_raw[i]

                # Convert to lists
                pres_list = pres_profile.tolist()
                temp_list = temp_profile.tolist()
                psal_list = psal_profile.tolist()

                cycle_data.append({
                    "profile_id": profile_id,
                    "wmo_float_id": platform_ids[i],
                    "cycle_id": int(ds["CYCLE_NUMBER"].values[i]),
                    "date_time_utc": julian_to_datetime(ds["JULD"].values[i]),
                    "latitude": float(ds["LATITUDE"].values[i]),
                    "longitude": float(ds["LONGITUDE"].values[i]),
                    "data_mode": current_data_mode,
                    "pres_mean_dbar": float(np.nanmean(pres_profile)),
                    "temp_mean_degC": float(np.nanmean(temp_profile)),
                    "psal_mean_psu": float(np.nanmean(psal_profile)),
                })

                full_arrays.append({
                    "profile_id": profile_id,
                    "temp_array": temp_list,
                    "psal_array": psal_list,
                    "pres_array": pres_list
                })

            return {"cycle_data": cycle_data, "full_arrays": full_arrays}

    except Exception as e:
        logging.error(f"Error during NetCDF parsing: {e}")
        return {"error": str(e)}

# --- Flask Route --- #
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

# --- Run --- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
