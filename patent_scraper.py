import requests
import pandas as pd
from datetime import datetime

def build_target_list():
    print("\n🚀 Booting up StartNerve Intelligence Lead Scraper...")
    print("📡 Connecting to official US FDA Database (OpenFDA)...")

    # FIXED: The FDA API requires the exact word "Prescription", not the number 1.
    url = "https://api.fda.gov/drug/drugsfda.json?search=products.marketing_status:Prescription&limit=200"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # SAFETY CATCH: If the FDA database returns a hidden error (like a typo in the URL)
        if 'error' in data:
            print(f"⚠️ FDA API Error: {data['error'].get('message', 'Unknown Error')}")
            return
            
        leads = []
        
        # Parse the messy JSON file into clean business data
        for result in data.get('results', []):
            sponsor_name = result.get('sponsor_name', 'Unknown Company')
            
            for product in result.get('products', []):
                brand_name = product.get('brand_name', 'Unknown Brand')
                
                for active_ingredient in product.get('active_ingredients', []):
                    api_name = active_ingredient.get('name', 'Unknown API')
                    
                    # Package it for Anwesha
                    leads.append({
                        "Target API (Chemical)": api_name.capitalize(),
                        "Brand Name": brand_name.capitalize(),
                        "Sponsor Company (Competitor/Lead)": sponsor_name.title(),
                        "Action Plan": "Find Indian Generic Manufacturers for this API"
                    })
                    
        # SAFETY CATCH 2: Check if leads is empty before trying to make a CSV
        if not leads:
            print("⚠️ No leads were found. The API returned 0 matching products.")
            return

        # Convert to a clean Pandas DataFrame and remove duplicates
        df = pd.DataFrame(leads).drop_duplicates(subset=["Target API (Chemical)"])
        
        # Save it directly as a CSV file in your folder
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"StartNerve_Hitlist_{date_str}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n✅ SUCCESS! Extracted {len(df)} High-Value Pharma Targets.")
        print(f"📁 Saved file as: {filename}")
        print("🎯 INSTRUCTIONS: Send this CSV to Anwesha to begin LinkedIn prospecting.")
        print("--------------------------------------------------\n")
        
    except Exception as e:
        print(f"⚠️ Error connecting to FDA Database: {e}")

if __name__ == "__main__":
    build_target_list()