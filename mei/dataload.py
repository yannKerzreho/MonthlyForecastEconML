import pandas as pd
import numpy as np
import dbnomics
from datetime import datetime
import argparse
import sys
import pandas_datareader.data as web

from config_data import DATA_MEI

# /!\ USA in first place
country_LIST = ["USA", "CAN", "GBR", "FRA", "DEU", "ITA", "ESP", "NLD", "AUT", "JPN", "KOR", "MEX"]

def period_to_month(period_str):
    try:
        return pd.to_datetime(str(period_str))
    except:
        return None
    
def config_to_fetch(cfg, country_list):
    dims = {
        "FREQUENCY": ["M"],
        "SUBJECT": [cfg["SUBJECT"]],
        "MEASURE": [cfg["MEASURE"]]
    }
    
    if isinstance(country_list, str):
        dims["LOCATION"] = [country_list]
    else:
        dims["LOCATION"] = country_list

    return dims

def preprocess_dataframe(df, key, geo_col="LOCATION", force_usa=False):
    df_clean = df[[geo_col, "original_period", "value"]].copy()
    df_clean = df_clean.rename(columns={
        geo_col: "country", 
        "original_period": "date", 
        "value": key
    })

    if force_usa:
        dates_ref = df_clean["date"].unique()
        df_usa_fix = pd.DataFrame({
                    "country": "USA",
                    "date": dates_ref,
                    key: 1.0
                })
        df_clean = pd.concat([df_clean, df_usa_fix], ignore_index=True)

    df_clean["date"] = df_clean["date"].apply(period_to_month)
    df_clean[key] = pd.to_numeric(df_clean[key], errors='coerce')   
    
    return df_clean

def build_macro_panel(table_configs, country_list=None, verbose=False):
    if country_list is None:
        country_list = country_LIST
    
    meta = {}
    partial_frames = []

    for key, cfg in table_configs.items():
        if verbose: print(f"Fetching {key} ({cfg['SUBJECT']})...")
        
        clsit = country_list
        force_usa = False

        if key == "Currency_conversions_usd" and "USA" in country_list:
            clsit = [c for c in country_list if c != "USA"]
            force_usa = True

        meta[key] = {
            "op": cfg.get("op"),
            "subject": cfg.get("SUBJECT"),
            "measure": cfg.get("MEASURE")
        }

        df = dbnomics.fetch_series(
            provider_code="OECD",
            dataset_code="MEI",
            dimensions=config_to_fetch(cfg, clsit)
            )

        if df is None or df.empty:
            if verbose: print(f"{key}: No data retrieved.")
            continue
        
        df_clean = preprocess_dataframe(df, key, force_usa=force_usa)        
        partial_frames.append(df_clean)

    if not partial_frames:
        print("No data retrieved for any variable.")
        return None, meta
    
    panel = partial_frames[0]
    
    for df_next in partial_frames[1:]:
        panel = pd.merge(panel, df_next, on=["country", "date"], how="outer")

    panel = panel.sort_values(["country", "date"]).reset_index(drop=True)
    
    return panel, meta


def apply_transformations(df, meta):
    if df is None or df.empty:
        return None

    df_transformed = df.copy()
    df_transformed = df_transformed.sort_values(["country", "date"])
    
    for col_name, info in meta.items():
        if col_name not in df_transformed.columns:
            continue
            
        operation = info.get("op")
        
        if not operation:
            continue
        
        try:
            if operation == "D":
                df_transformed[col_name] = df_transformed.groupby("country")[col_name].diff()
                
            elif operation == "Dlog":
                log_series = np.log(df_transformed[col_name])
                df_transformed[col_name] = log_series.groupby(df_transformed["country"]).diff()

        except Exception as e:
            print(f"Error {col_name}: {e}")
            
    return df_transformed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Main Economic Indicators from OCDE with DBnomics."
    )
    
    parser.add_argument(
        "-p", "--path", 
        type=str, 
        default="macro_data.csv", 
        help="Path to save file. Default: 'mei/macro_data.csv'"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Print detailed logs during execution."
    )

    args = parser.parse_args()
    
    try:
        df_panel, metadata = build_macro_panel(DATA_MEI, verbose=args.verbose)

        if df_panel is not None and not df_panel.empty:
            df_final = apply_transformations(df_panel, metadata)
            output_path = args.path

            if output_path.endswith(".xlsx") or output_path.endswith(".xls"):
                try:
                    df_final.to_excel(output_path, index=False)
                    print(f"File written at {output_path}")
                except ImportError:
                    print("Error : To save in xlsx, install 'openpyxl' (pip install openpyxl).")
                    print("CSV instead...")
                    df_final.to_csv(output_path.replace(".xlsx", ".csv"), index=False)
            else:
                df_final.to_csv(output_path, index=False)
                print(f"File written at {output_path}")            
        else:
            print("DataFrame is empty. No file generated.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n Interruption by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)