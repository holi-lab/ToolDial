import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random_template_generation(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint serves as the main entry point or home page for the app. The api generates a random set of data for the resume template using the a function. Then, it renders the template file, passing the generated data as a variable named data to be used within the HTML template. The rendered HTML page displays the randomly generated resume template with the provided data. [https://youtu.be/ZDX5XtbXomE](url)"
    
    """
    url = f"https://random-gradient-resume-template-generator.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-gradient-resume-template-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_generated_template_zip(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is responsible for generating a ZIP file containing the randomly generated resume template and its associated assets (CSS and JavaScript files)."
    
    """
    url = f"https://random-gradient-resume-template-generator.p.rapidapi.com/download"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-gradient-resume-template-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

