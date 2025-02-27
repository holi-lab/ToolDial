import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pst_password_remover(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can reset the password of Outlook PST file using MailsDaddy PST Password Recovery Tool. This tool resets the password of PST file in just a few clicks. The special feature of this tool is that you can reset the new password even without the old password. This tool cracks passwords without damaging the data. It extracts PST passwords securely in one click. The tool supports both ANSI and Unicode PST formats. And also supports all MS Outlook 2019,2016,2013,2010,2007 etc versions.
		VISIT FOR MORE INFORMATION:-[https://www.mailsdaddy.com/pst-password-remover/](url)"
    
    """
    url = f"https://pst-password-remover.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pst-password-remover.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

