import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchjobs(location: str, keyword: str, offset: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "offset = 0 (starting of the page, it must be increased by 10 to achieve pagination)
		keyword = python (it can be any search keyword for ex: job title or skill title)
		location = Helsinki (For now this API gets data for Finland Indeed. You can enter a specific city or state.)
		This will provide you with a list of 15 jobs in the page, by default a single page can have a max of 15 jobs posting only. In order to get all the data using pagination you need to keep increasing the count of offset by 10.
		
		You will get the following fields using this API.
		
		'position'
		'company_name'
		'job_title'
		'job_location'
		'salary'
		'date'
		'job_url'
		'urgently_hiring'
		'multiple_hiring'
		'company_rating'
		'company_reviews'
		'company_review_link'
		'company_logo_url'
		'page_number'"
    
    """
    url = f"https://indeed-jobs-api-finland.p.rapidapi.com/indeed-fi/"
    querystring = {'location': location, 'keyword': keyword, 'offset': offset, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indeed-jobs-api-finland.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

