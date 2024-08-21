from zenrows import ZenRowsClient
import os


# zen rows client
zenrows_client = ZenRowsClient(apikey=os.getenv("ZENROWS_API_KEY"), concurrency=1)
