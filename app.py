import os
import tempfile
import pandas as pd
import xarray as xr
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import logging
import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- Helper Function for Argo Time Conversion ---
def julian_to_datetime(juld_days):
    """Converts Julian Day (days since 1950-01-01) to ISO datetime string."""
    if pd.isna(juld_days):
        return None
    try:
        ref_date = datetime(1950, 1, 1, 0, 0, 0)
        dt = ref_date + timedelta(days=float(juld_days))
        return dt.isoformat()
    except Exception:
        return None

# --- Core NetCDF Parsing Logic ---
def parse_argo_file(file_path):
    """Reads a multi-profile NetCDF file and extracts data into a structured list."""
    try:
        with xr.open_dataset(file_path, decode_times=False) as ds:

            # --- 1. Extract Global Float Metadata ---
            if "PLATFORM_NUMBER" in ds:
                try:
                    wmo_float_id = "".join(ds["PLATFORM_NUMBER"].values.astype(str)).strip()
                except Exception:
                    wmo_float_id = str(ds["PLATFORM_NUMBER"].values.flatten()[0]).strip()
            else:
                wmo_float_id = "UNKNOWN"

            # --- 2. Extract Cycle Summary Data ---
            required_meta_vars = ["CYCLE_NUMBER", "JULD", "LATITUDE", "LONGITUDE", "DATA_MODE"]
            for var in required_meta_vars:
                if var not in ds:
                    raise ValueError(f"Missing expected variable: {var}")

            cycle_data_df = pd.DataFrame({
                "cycle_id": ds["CYCLE_NUMBER"].values,
                "juld_days": ds["JULD"].values,
                "latitude": ds["LATITUDE"].values,
                "longitude": ds["LONGITUDE"].values,
                "data_mode": ds["DATA_MODE"].values.astype(str),
            })

            cycle_data_df["date_time_utc"] = cycle_data_df["juld_days"].apply(julian_to_datetime)

            # --- 3. Extract High-Resolution Profile Data ---
            def safe_stack(var_name):
                if var_name not in ds:
                    logging.warning(f"{var_name} not found in dataset, filling with NaNs")
                    return pd.DataFrame()
                try:
                    return ds[var_name].stack(points=("N_PROF", "N_LEVELS")).to_dataframe().reset_index()
                except Exception as e:
                    logging.error(f"Error stacking {var_name}: {e}")
                    return pd.DataFrame()

            pressures_df = safe_stack("PRES")
            temperatures_df = safe_stack("TEMP_ADJUSTED")
            salinities_df = safe_stack("PSAL_ADJUSTED")

            if pressures_df.empty:
                raise ValueError("No pressure data available in file.")

            profile_df = pressures_df
            if not temperatures_df.empty:
                profile_df = profile_df.merge(temperatures_df, on=["N_PROF", "N_LEVELS"], how="left")
            if not salinities_df.empty:
                profile_df = profile_df.merge(salinities_df, on=["N_PROF", "N_LEVELS"], how="left")

            # Merge with cycle metadata (N_PROF links profiles to cycles)
            profile_df = profile_df.merge(cycle_data_df, left_on="N_PROF", right_index=True, how="left")

            # --- 4. Cleanup and Formatting ---
            # Filter out fill values (< -9000 often used)
            if "PRES" in profile_df:
                profile_df = profile_df[profile_df["PRES"] > -9000].copy()

            profile_df.rename(columns={
                "N_PROF": "prof_index",
                "PRES": "pressure_dbar",
                "TEMP_ADJUSTED": "temperature_c",
                "PSAL_ADJUSTED": "salinity_psu"
            }, inplace=True)

            profile_df["wmo_float_id"] = wmo_float_id

            final_columns = [
                "wmo_float_id", "cycle_id", "date_time_utc", "latitude", "longitude",
                "pressure_dbar", "temperature_c", "salinity_psu", "data_mode"
            ]
            for col in final_columns:
                if col not in profile_df:
                    profile_df[col] = np.nan  # ensure column exists

            return profile_df[final_columns].to_dict("records")

    except Exception as e:
        app.logger.error(f"Error during NetCDF parsing: {e}")
        raise ValueError(f"Failed to process NetCDF file: {e}")

# --- Flask API Endpoint ---
@app.route("/parse_argo", methods=["POST"])
def parse_argo_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".nc"):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
                file.save(tmp_file.name)
                temp_path = tmp_file.name

            app.logger.info(f"Processing temporary file: {temp_path}")
            parsed_data = parse_argo_file(temp_path)

            return jsonify({
                "status": "success",
                "message": f"Successfully processed {len(parsed_data)} measurement points.",
                "total_rows": len(parsed_data),
                "profile_data": parsed_data,
            }), 200

        except ValueError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                app.logger.info(f"Cleaned up temporary file: {temp_path}")
    else:
        return jsonify({"error": "Invalid file type. Must be a .nc file"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("PORT", 5000))
