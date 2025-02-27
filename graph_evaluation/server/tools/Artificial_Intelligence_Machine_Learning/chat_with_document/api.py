import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_session_status_api(sessionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API let you check the session status by passing in sessionId. The supported status are:
		pending - The session is just created
		downloading - Backend job is just available and pick up your URL and downloading the documents
		downloaded - Your document just successfully downloaded and store in our cloud storage
		ingesting - Backend job start ingesting your document into embedding
		ingested - Backend job successfully ingest your document and you are ready to evaluate (start the chat interaction)
		error_ingested - There are error on server side to ingest your document."
    
    """
    url = f"https://chat-with-document.p.rapidapi.com/getsessionstatus"
    querystring = {'sessionId': sessionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chat-with-document.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

