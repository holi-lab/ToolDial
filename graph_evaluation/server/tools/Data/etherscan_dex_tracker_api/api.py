import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def date_range(end_date: str, start_date: str, contract_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The **/date_range/** API endpoint enables users to retrieve a range of blocks associated with a specific contract address. This endpoint returns a maximum of 5000 blocks within the specified date range.
		
		The endpoint accepts three parameters: **contract_address**, **start_date**, and **end_date**, which define the date range for the desired blocks.
		
		Developers and users who want to analyze or extract data from a particular contract within a specific date range will find this endpoint particularly useful. By providing the contract address along with the starting and ending dates, users can efficiently retrieve a focused subset of blocks associated with the contract, facilitating targeted data retrieval."
    end_date: Date format: mm/dd/yyyy
        start_date: Date format: mm/dd/yyyy
        
    """
    url = f"https://etherscan-dex-tracker-api.p.rapidapi.com/date_range/"
    querystring = {'end_date': end_date, 'start_date': start_date, 'contract_address': contract_address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etherscan-dex-tracker-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_range(block_number_from: str, block_number_to: str, contract_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The **/block_range/** API endpoint allows users to retrieve a range of blocks associated with a specific contract address. The endpoint has a maximum limit of 5000 blocks for the block range.
		
		The endpoint accepts three parameters: **contract_address**, **block_number_from**, and **block_number_to**, which specify the block numbers between which the desired blocks are to be retrieved. However, the block range must not exceed 5000 blocks.
		
		This endpoint is particularly useful for developers or users who wish to analyze or extract data from a specific contract within a defined block range. By providing the contract address along with the starting and ending block numbers, users can efficiently retrieve a subset of blocks associated with the contract, facilitating focused data retrieval."
    
    """
    url = f"https://etherscan-dex-tracker-api.p.rapidapi.com/block_range/"
    querystring = {'block_number_from': block_number_from, 'block_number_to': block_number_to, 'contract_address': contract_address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etherscan-dex-tracker-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

