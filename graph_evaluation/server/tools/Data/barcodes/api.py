import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_product_details(query: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It returns product results if you enter a search term, and returns product details if you enter a barcode number."
    query: **Barcode** or **search term**
        page: If you want search results - **page number**
If you are querying for a barcode, leave it blank.

If you leave it blank (if you are querying for a search term), the first page will be returned.
        
    """
    url = f"https://barcodes1.p.rapidapi.com/"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcodes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

