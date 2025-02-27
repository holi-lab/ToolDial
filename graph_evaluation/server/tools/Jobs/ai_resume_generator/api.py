import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_document_cover_letter(jobdescription: str='Example Co. is looking for a Full Stack Web Developer to maintain and improve our custom web application. The solutions will require both frontend and backend skills. Our web application runs on a LEMP stack. LEMP stack experience is a bonus, but not required.', degree: str='Information Systems', university: str='Michigan State University', skills: str='Networking, Web Development', jobtitle: str='Full Stack Web Developer', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint that will generate a large text response containing a Cover Letter document using the provided information"
    
    """
    url = f"https://ai-resume-generator.p.rapidapi.com/Documents/GenerateCoverLetter"
    querystring = {}
    if jobdescription:
        querystring['jobDescription'] = jobdescription
    if degree:
        querystring['degree'] = degree
    if university:
        querystring['university'] = university
    if skills:
        querystring['skills'] = skills
    if jobtitle:
        querystring['jobTitle'] = jobtitle
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-resume-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_document_resume(degree: str='Information Systems', skills: str='Networking, Web Development', university: str='Michigan State University', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint that will generate a large text response containing a Resume document using the provided information"
    
    """
    url = f"https://ai-resume-generator.p.rapidapi.com/Documents/GenerateResume"
    querystring = {}
    if degree:
        querystring['degree'] = degree
    if skills:
        querystring['skills'] = skills
    if university:
        querystring['university'] = university
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-resume-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

