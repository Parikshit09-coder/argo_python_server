import os
import json
import tempfile
import pandas as pd
import xarray as xr
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- Helper Function for Argo Time Conversion ---

def julian_to_datetime(juld_days):
    """Converts Julian Day (days since 1950-01-01) to ISO datetime string."""
    if pd.isna(juld_days):
        return None

    # Argo reference date: 1950-01-01 00:00:00 UTC
    ref_date = datetime(1950, 1, 1, 0, 0, 0, tzinfo=None)
    dt = ref_date + timedelta(days=float(juld_days))
    return dt.isoformat()

# --- Core NetCDF Parsing Logic ---

def parse_argo_file(file_path):
    """Reads a multi-profile NetCDF file and extracts data into a structured list."""
    try:
        # Open the NetCDF file using xarray
        with xr.open_dataset(file_path, decode_times=False) as ds:
            
            # 1. Extract Global Float Metadata (Static Data)
            wmo_float_id = str(ds['PLATFORM_NUMBER'].values.item()).strip()
            
            # 2. Extract Cycle Summary Data
            cycle_data_df = pd.DataFrame({
                'cycle_id': ds['CYCLE_NUMBER'].values,
                'juld_days': ds['JULD'].values,
                'latitude': ds['LATITUDE'].values,
                'longitude': ds['LONGITUDE'].values,
                'data_mode': ds['DATA_MODE'].values.astype(str)
            })
            
            # Convert Julian days to ISO datetime strings
            cycle_data_df['date_time_utc'] = cycle_data_df['juld_days'].apply(julian_to_datetime)
            
            # 3. Extract High-Resolution Profile Data
            # Use .stack() to flatten the multi-dimensional arrays efficiently
            pressures_df = ds['PRES'].stack(points=('N_PROF', 'N_LEVELS')).to_dataframe().reset_index()
            temperatures_df = ds['TEMP_ADJUSTED'].stack(points=('N_PROF', 'N_LEVELS')).to_dataframe()
            salinities_df = ds['PSAL_ADJUSTED'].stack(points=('N_PROF', 'N_LEVELS')).to_dataframe()
            
            # Merge all three dataframes
            profile_df = pressures_df.merge(temperatures_df, on=['N_PROF', 'N_LEVELS'])
            profile_df = profile_df.merge(salinities_df, on=['N_PROF', 'N_LEVELS'])
            
            # Merge with cycle metadata
            final_df = profile_df.merge(cycle_data_df, left_on='N_PROF', right_index=True)

            # Filter out invalid values (e.g., fill values are typically large negative numbers)
            final_df = final_df[final_df['PRES'] > -9000].copy()

            # 4. Final Formatting and Selection of Columns
            final_df.rename(columns={
                'N_PROF': 'prof_index',
                'PRES': 'pressure_dbar',
                'TEMP_ADJUSTED': 'temperature_c',
                'PSAL_ADJUSTED': 'salinity_psu'
            }, inplace=True)

            final_df['wmo_float_id'] = wmo_float_id

            # Select and order the final columns
            final_columns = [
                'wmo_float_id', 'cycle_id', 'date_time_utc', 'latitude', 'longitude', 
                'pressure_dbar', 'temperature_c', 'salinity_psu', 'data_mode'
            ]
            
            return final_df[final_columns].to_dict('records')

    except Exception as e:
        app.logger.error(f"Error during NetCDF parsing: {e}")
        raise ValueError(f"Failed to process NetCDF file: {e}")

# --- Flask API Endpoint ---

@app.route('/parse_argo', methods=['POST'])
def parse_argo_endpoint():
    """
    Receives a POST request with the .nc file, parses it, and returns JSON data.
    """
    if 'file' not in request.files:
        app.logger.warning("No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        app.logger.warning("No selected file")
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.nc'):
        try:
            # Use tempfile to securely create a temporary path
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                file.save(tmp_file.name)
                temp_path = tmp_file.name
            
            app.logger.info(f"Processing temporary file: {temp_path}")
            
            # Process the file and return data
            parsed_data = parse_argo_file(temp_path)
            
            return jsonify({
                "status": "success",
                "message": f"Successfully processed {len(parsed_data)} measurement points.",
                "total_rows": len(parsed_data),
                "profile_data": parsed_data
            }), 200

        except ValueError as e:
            app.logger.error(f"Parsing error: {str(e)}")
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            app.logger.error(f"Unexpected error: {str(e)}")
            return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                app.logger.info(f"Cleaned up temporary file: {temp_path}")
    else:
        return jsonify({"error": "Invalid file type. Must be a .nc file"}), 400

if __name__ == '__main__':
    # Use Gunicorn or similar WSGI server for production on Render
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
