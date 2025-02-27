import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def business_days_work_days_calculator_api(start_date: str, work_days: int, state: str, options: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Workdays Calculation API Documentation
		
		The Workdays Calculation API allows you to calculate the date after a given number of workdays, taking into account weekends and holidays. By specifying the country code, number of workdays, start date, and options, you can retrieve the ISO date of the day after the workdays have passed.
		
		
		## Request Parameters
		
		The API expects the following GET parameters:
		
		1. `state` (required): The country code for which the workdays are calculated. Please refer to the [list of country codes](https://www.nationsonline.org/oneworld/international-calling-codes.htm) to find the appropriate code for your country.
		
		2. `work_days` (required): The number of workdays to calculate. This indicates how many workdays should be skipped before determining the result.
		
		3. `start_date` (required): The starting date from which to begin counting the workdays. The format of the start date should be `DD/MM/YYYY`.
		
		4. `options` (optional): The options for skipping specific days. This parameter accepts the following values:
		   - `0`: Skip holidays and weekends.
		   - `1`: Skip only weekends.
		   - `2`: Skip only holidays.
		
		## Response
		
		The API response will be a string representing the ISO date of the day after the specified workdays have passed.
		
		## Example
		
		Given `start_date` of 19/05/2023, which is Friday, and `work_days` of 2, 
		the returned result will be 23/05/2023. Here's the breakdown of the calculation:
		
		- 19/5-20/5: Weekend (Saturday and Sunday)
		- 21/5: First workday
		- 22/5: Second workday
		
		Therefore, 23/5 is the day after the specified workdays have passed"
    
    """
    url = f"https://business-days-work-days-calculator.p.rapidapi.com/api/v1/get_result"
    querystring = {'start_date': start_date, 'work_days': work_days, 'state': state, 'options': options, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-days-work-days-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

