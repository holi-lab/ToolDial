import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def which_doctor(specialist: bool, condition: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Which doctor to consult based on the disease / symptoms ?**
		
		Input: We will require two input points:
		
		1. Condition: This input is entered as a string. You need to enter either the name of the disease eg: "Diabetes" or some specific symptoms eg: excessive vomiting.
		2. Specialist: This input is entered as a boolean. Enter "true" if a specialist consult is required. Enter false in which case the response may or may not include a specialist.
		
		Output:
		
		If your input was "Diabetes" and Specialist was "true"
		The output is returned as a string which will be something like this:
		
		"You should consult with an endocrinologist, who specializes in diabetes and other endocrine disorders"
		
		If your input was "vomiting" and Specialist was "false"
		
		"It is best to consult your primary care physician for vomiting. Depending on the cause of your vomiting, your doctor may refer you to a specialist such as a gastroenterologist or an infectious disease specialist""
    
    """
    url = f"https://doctorgpt.p.rapidapi.com/doctorgpt"
    querystring = {'specialist': specialist, 'condition': condition, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "doctorgpt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

