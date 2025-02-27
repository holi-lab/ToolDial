import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_instruction_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a specific instruction.
		
		`URL: /instructions/<instruction_id>/`
		
		**Method:** GET
		
		**Auth required:** Yes
		
		**Path parameters:**
		
		**instruction_id: **The ID of the instruction to retrieve.
		
		**Success response:
		**
		
		```
		Code: 200 OK
		Content: { "id": 1, "description": "Summarize the text", "tones": [ { "id": 1, "name": "professional" } ], "category": { "id": 1, "name": "marketing" } }
		```
		
		**Error response:**
		
		```
		Code: 404 NOT FOUND
		Content: { "error": "Instruction not found" }
		```"
    
    """
    url = f"https://nexia2.p.rapidapi.com/instructions/{is_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nexia2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_instruction(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for instructions by description, category, and/or tones.
		
		`URL: /instructions/search/`
		
		**Method:** GET
		
		**Auth required: **Yes
		
		**URL parameters:**
		
		**description: **A string to search for in the description field.
		**category:** A string to search for in the category name.
		**tones:** A list of tone names to filter by. Use multiple times to filter by multiple tones.
		
		**Sample request:**
		
		`GET /instructions/search/?description=summarize&category=marketing&tones=professional`
		
		**Success response:**
		
		```
		**Code: **200 OK
		**Content:** [ { "id": 1, "description": "Summarize the text", "tones": [ { "id": 1, "name": "professional" } ], "category": { "id": 1, "name": "marketing" } } ]
		
		```
		
		
		Error response:
		
		Code: 400 BAD REQUEST
		Content: { "errors": { "tones": [ "This field is required." ] } }"
    
    """
    url = f"https://nexia2.p.rapidapi.com/instructions/search/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nexia2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

