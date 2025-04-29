import requests
import time
from collections import deque

#Consts
REQUEST_WINDOW = 60 
MAX_REQUESTS_PER_10_SECONDS = 5
SHORT_TERM_WINDOW = 10 
MAX_RETRIES = 5

#track timestamps of the last 5 requests using a queue
recent_requests = deque(maxlen=MAX_REQUESTS_PER_10_SECONDS)

def handle_api_request(url, max_retries=MAX_RETRIES):
    """
    Func to efficiently and automatically handle TotalCorners API rate limiting of:
    - 5 requests per 10 seconds
    - and 30 requests per min.
    """
    retries = 0

    while retries < max_retries:
        while len(recent_requests) >= MAX_REQUESTS_PER_10_SECONDS:
            time_since_oldest= time.time()-recent_requests[0]
            if time_since_oldest<SHORT_TERM_WINDOW:
                wait_time = SHORT_TERM_WINDOW -time_since_oldest
                print(f"🚦 5 requests in 10 seconds limit reached. Waiting {wait_time:.2f} seconds...")
                
                time.sleep(wait_time)
            else:
                break

        response = requests.get(url)
        recent_requests.append(time.time())

        #Get rate limit headers:
        rate_limit_remaining = int(response.headers.get("X-Rate-Limit-Remaining", 1))
        rate_limit_reset = int(response.headers.get("X-Rate-Limit-Reset", time.time() + REQUEST_WINDOW))

        if response.status_code == 200:
            data = response.json()
            if data.get("success") == 1:
                #if near the 30 requests/min limit, wait until reset
                if rate_limit_remaining == 0:
                    wait_time = max(1, rate_limit_reset-time.time())  # Ensure positive wait time
                    print(f"🚦 30 requests per minute limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

                return data #Successful API call!!

            elif data.get("error",{}).get("code") =="TOO_MANY_REQUEST":
                wait_time= max(1,rate_limit_reset-time.time())  # Ensure positive wait time
                print(f"⚠️ API limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time) 
                retries += 1  

            else:
                print(f"❌ API Error:{data.get('error', {}).get('message')}")
                return None 

        else:
            print(f"❌ HTTP Error {response.status_code}:{response.text}")
            return None 

    print(f"❌ Failed after {max_retries} retries.")
    return None
 