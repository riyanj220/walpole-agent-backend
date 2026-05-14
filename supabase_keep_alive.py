import os
import requests
from datetime import datetime

# Using your exact environment variable names
# os.getenv: Retrieves the value of a specific environment variable from the system.
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_SERVICE_KEY")

headers = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json"
}

def keep_alive():
    if not URL or not KEY:
        print("Error: SUPABASE_URL or SUPABASE_SERVICE_KEY is not set.")
        return

    # 1. INSERT a record to the 'pings' table
    # requests.post: Sends an HTTP POST request to your Supabase REST API to create a record.
    res_insert = requests.post(f"{URL}/rest/v1/pings", headers=headers, json={"created_at": "now()"})
    
    # 2. SELECT to verify activity
    # requests.get: Fetches data from the table to simulate real user read activity.
    requests.get(f"{URL}/rest/v1/pings?limit=1", headers=headers)
    
    # 3. DELETE old records to keep the table empty
    # requests.delete: Cleans up the table so you stay within your free-tier storage limits.
    # requests.delete(f"{URL}/rest/v1/pings?id=gt.0", headers=headers)

    if res_insert.status_code in [200, 201]:
        print(f"Successfully pinged Supabase at {datetime.now()}")
    else:
        print(f"Failed to ping: {res_insert.text}")

if __name__ == "__main__":
    keep_alive()