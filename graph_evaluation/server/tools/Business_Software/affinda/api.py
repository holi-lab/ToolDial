import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def affinda_resume_parser(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Affinda Resume Parser is a machine learning software that uses NLP (Natural Language Processing) to extract more than 100 fields from each resume. It also organizes the resumes into searchable file formats. Fields that are extracted through resume parser are personal details, work experience, education, certifications, skills, summary, language, referees, publications, etc. We offer accurate resume parsing for ATS (Applicant Tracking Systems), Job boards, Staffing services, HR Teams, Recruiters, and more and can work with you to adapt our technology to your parsing needs."
    
    """
    url = f"https://affinda.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "affinda.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

