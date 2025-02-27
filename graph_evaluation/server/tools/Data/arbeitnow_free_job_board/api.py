import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def free_job_postings_api(content_type: str='application/json', visa_sponsorship: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Built in Berlin, Arbeitnow helps companies hire candidates with visa sponsorship / four day work week / remote opportunities.
		
		Four day work weeks - Companies offering better work life balance with a shorter work week results in improved productivity and helps retain candidates. Get your company listed on the [4 day work week jobs](https://arbeitnow.com/4-day-work-week-jobs) page
		
		Hiring without Whiteboard - No trivia questions or stress inducing whiteboard interviews? We will include you in our popular section among tech & software developers - [hiring without whiteboard.](https://arbeitnow.com/hiring-without-whiteboard)
		
		[Jobs with Salary / Compensation](https://arbeitnow.com/jobs-with-salary) information helps bring transparency to candidates on what they can expect from an offer."
    
    """
    url = f"https://arbeitnow-free-job-board.p.rapidapi.com/api/job-board-api"
    querystring = {}
    if content_type:
        querystring['Content-Type'] = content_type
    if visa_sponsorship:
        querystring['visa_sponsorship'] = visa_sponsorship
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arbeitnow-free-job-board.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

