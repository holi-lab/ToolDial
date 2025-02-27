import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_result(job_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will show the stage the job is currently in, and if the job is completed, the verification results.
		Response properties description:
		
		```
		{
		  "job_id": "xxx",
		  "created_at": "2023-04-04T06:16:51.454000", // the time the user made the request
		  "started_at": "2023-04-04T06:17:31.863000", // the time the job started
		  "finished_at": "2023-04-04T06:17:37.002000", // the time the job finished
		  "total_emails": 2, // number of emails we found in the list
		  "total_processable": 2, // number of unique emails
		  "percent_complete": 100, // overall progress of the job's verification
		  "status": "complete", // what stage the job is currently in
		  "results": { // verification status of the emails
		    "jane@example.com": "valid",
		    "john@example.com": "invalid"
		  }
		}
		```
		
		Job status:
		- `waiting` - the job has not yet run and is waiting to be started.
		- `queued` - the job has been queued, this is because we're either too busy or you have too many active jobs.
		- `running` - the job is currently running.
		- `complete` - the job has completed verification and the results are ready for download.
		
		Possible verification statuses:
		- `valid` - the email address is valid and unlikely to bounce.
		- `invalid` - the email address is invalid and will bounce.
		- `unknown` - there was an error that happened during verification, consequently the validity of the email address is unknown. It may or may not bounce, if you send an email to it with an actual mail server.
		- `catchall` - the email address is unlikely to bounce. The mail server has been configured to accept any emails addressed to any recipients on the email domain."
    
    """
    url = f"https://leadbook-email-verification-list-cleaning1.p.rapidapi.com/v1/result"
    querystring = {'job_id': job_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "leadbook-email-verification-list-cleaning1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

