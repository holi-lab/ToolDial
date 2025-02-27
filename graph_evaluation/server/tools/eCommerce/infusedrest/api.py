import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def infusedopportunityresults(infusionsoft_app_name: str, infusionsoft_api_key: str, display_mode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A single Infusionsoft opportunity report that will show Opportunity Name, Contact Name, Company, Owner, Stage, Date Created, and Order Revenue."
    infusionsoft_app_name: This Parameter is the Name of your Infusionsoft App Name. You should be able to get it from the Base URL of your Infusionsoft App.
For example, if your Infusionsoft App URL is
http://<TikTik>.infusionsoft.com. then this Parameter Value would be \"TikTik\".
        infusionsoft_api_key: - Login to your Infusionsoft Application.
- Navigate to Admin->Settings->Application(Left side Menu)
- Scroll down and you should see \"Encrypted Key:\". Thats the Value you need to pass this for API interaction.
- 

**Note:**

-   If this \"Encrypted Key:\" Value is Null, then Enter any Secret Code in field \"API Passphrase:\" and click Save.
-   On Page reload, the \"Encrypted Key:\" field would get populated.
-   Thats the value you need to pass for this Parameter.
        display_mode: You can pass values like \"HTML\" or \"JSON\" for this parameter.
HTML: If passed as \"HTML\", the Rest API would return a fully presentable HTML Table with the Opportunity Data Set as Report.
JSON: If passed as \"JSON\",  the Rest API would return Opportunity data as JSON String which in-turn can be consumed in any client application. 
        
    """
    url = f"https://infusedrest.p.rapidapi.com/OpportunityData.php"
    querystring = {'INFUSIONSOFT_APP_NAME': infusionsoft_app_name, 'INFUSIONSOFT_API_KEY': infusionsoft_api_key, }
    if display_mode:
        querystring['DISPLAY_MODE'] = display_mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "infusedrest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

