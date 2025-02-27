import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def diagnosticscodelist(diagnosticscode: str, diagnosticsname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Diagnostics Code Listِ  A lightweight API based on ICD-10-CM (International Classification of Diseases, Tenth Revision, Clinical Modification) that displays medical diagnoses data using the diagnostic code, the diagnosis name, or symptoms, which can be used by health care providers ( health information managers, nurses and other healthcare professionals ) clinics and hospitals in their own systems or be a part of prognosis and diagnosis systems, Search can be done using diagnostics Codes or the diagnosis name, symptoms, and procedure names.  ( result is JSON you can use it in your site or your own system )  DiagnosticsCode : String  DiagnosticsName : String"
    diagnosticscode: Diagnostics Code
        diagnosticsname: Diagnostics Name
        
    """
    url = f"https://diagnostics-code-list.p.rapidapi.com/{diagnosticscode}/{diagnosticsname}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diagnostics-code-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

