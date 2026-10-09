import os
import requests
import re

from dotenv import load_dotenv

load_dotenv()

class IndianKanoonClient:
    BASE_URL = "https://api.indiankanoon.org"

    def __init__(self):
        self.api_key = os.getenv("INDIAN_KANOON_API_KEY")

        if not self.api_key:
            raise ValueError("API KEY Not Provided")
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json",
        }

    def search(self, query, page=0):
        url = f"{self.BASE_URL}/search/"
        params = {
            "formInput": query,
            "pagenum": page,
        }
        response = requests.post(
            url,
            headers=self.headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    
    def get_document(self, document_id):
        url = f"{self.BASE_URL}/doc/{document_id}/"

        response = requests.post(
            url,
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()