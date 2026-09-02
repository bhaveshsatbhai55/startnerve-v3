import pandas as pd
import urllib.parse
from datetime import datetime
import os

def build_indian_hitlist():
    print("\n🚀 Booting up StartNerve Direct-Injection Lead Generator...")
    print("📡 Bypassing Search Engines... Connecting directly to IndiaMART routing...")

    # 1. Load the US FDA Hitlist
    us_file = "StartNerve_Hitlist_2026-02-21.csv"
    
    if not os.path.exists(us_file):
        print(f"⚠️ Error: Could not find {us_file}. Please run the FDA scraper first.")
        return

    print("📁 Loaded US FDA Hitlist. Extracting Target APIs...")
    fda_df = pd.read_csv(us_file)
    
    # Extract unique APIs (We can do ALL of them now since there is no rate limit!)
    apis = fda_df['Target API (Chemical)'].dropna().unique()
    
    indian_leads = []
    
    print(f"⚡ Generating direct database links for {len(apis)} APIs...\n")
    
    # 2. Dynamically construct the exact search URLs
    for api in apis:
        # Create a URL-friendly search query (e.g., "Ondansetron API")
        query = f"{api} API"
        encoded_query = urllib.parse.quote_plus(query)
        
        # Build the exact URLs that Arnav would normally have to search for manually
        indiamart_url = f"https://dir.indiamart.com/search.mp?ss={encoded_query}"
        tradeindia_url = f"https://www.tradeindia.com/search.html?keyword={encoded_query}"
        
        indian_leads.append({
            "Target API": api.capitalize(),
            "IndiaMART Direct Link": indiamart_url,
            "TradeIndia Direct Link": tradeindia_url,
            "Assigned To": "Arnav",
            "Action Plan": "Click link -> Find top 3 factory owners -> Pitch Dossier"
        })

    # 3. Save the results into an actionable sales sheet
    leads_df = pd.DataFrame(indian_leads)
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"StartNerve_Indian_Targets_{date_str}.csv"
    leads_df.to_csv(filename, index=False)
    
    print(f"\n✅ SUCCESS! Generated {len(leads_df)} direct supplier database routing links.")
    print(f"📁 Saved file as: {filename}")
    print("🎯 INSTRUCTIONS: Arnav doesn't need to search Google. He just clicks the links in this CSV and the factories appear.")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    build_indian_hitlist()