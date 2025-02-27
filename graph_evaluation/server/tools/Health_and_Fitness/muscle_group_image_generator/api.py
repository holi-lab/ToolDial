import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_base_image(transparentbackground: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the base image without any highlighted muscle groups"
    transparentbackground: Set the image background to transparent with a value of 1. The default is 0 and a white background color.
        
    """
    url = f"https://muscle-group-image-generator.p.rapidapi.com/getBaseImage"
    querystring = {}
    if transparentbackground:
        querystring['transparentBackground'] = transparentbackground
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "muscle-group-image-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_individual_color_image_set_color_for_each_muscle(musclegroups: str, colors: str, transparentbackground: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create an image where each muscle can have a different highlight color. Other than in the other endpoints you have to set the colors as hex values."
    musclegroups: A list of the muscle groups separated by comma that should be highlighted. You can see the available muscle groups via the /getMuscleGroups endpoints
        colors: A list of colors as hex strings that should have a length less or equal than the muscleGroups list. 
If the length of the colors list is shorter each muscle that have no corresponding color will be highlighted in the last color of the list. Example:

muscleGroups: biceps,triceps,shoulders
colors: f00,0f0
=> Biceps will be f00 and triceps, shoulders will be 0f0  
        transparentbackground: When set to 1 the background of the image will be transparent. The default value is 0 (white background)
        
    """
    url = f"https://muscle-group-image-generator.p.rapidapi.com/getIndividualColorImage"
    querystring = {'muscleGroups': musclegroups, 'colors': colors, }
    if transparentbackground:
        querystring['transparentBackground'] = transparentbackground
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "muscle-group-image-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_available_muscle_groups(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available muscle groups as a list"
    
    """
    url = f"https://muscle-group-image-generator.p.rapidapi.com/getMuscleGroups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "muscle-group-image-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_single_color_image_just_primary_musclegroups(musclegroups: str, transparentbackground: int=0, color: str='200,100,80', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an image where all requested muscleGroups are highlighted."
    musclegroups: A list of musclegroups that should be highlighted. Get a list of possible muscle groups with the /getMuscleGroups endpoint.
        transparentbackground: When set to 1 the background of the image will be transparent. Default value is 0.
        color: The color that the highlighted musclegroups should have. The value of the parameter should contain the rgb values as a string like \\\\\\\"red,green,blue\\\\\\\".
        
    """
    url = f"https://muscle-group-image-generator.p.rapidapi.com/getImage"
    querystring = {'muscleGroups': musclegroups, }
    if transparentbackground:
        querystring['transparentBackground'] = transparentbackground
    if color:
        querystring['color'] = color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "muscle-group-image-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_dual_color_image_primary_and_secondary_musclegroups(secondarymusclegroups: str, primarycolor: str, primarymusclegroups: str, secondarycolor: str, transparentbackground: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an image where all requested primary and secondary muscleGroups are highlighted in their corresponding color. If you only need one highlight color you can use the "Get Image" endpoint."
    secondarymusclegroups: Example: \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"biceps,triceps,chest\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\". See possible musclegroups via the /getMuscleGroups endpoint.
        primarycolor: The color that the highlighted **primary **musclegroups should have. The value of the parameter should contain the rgb values as a string like \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"red,green,blue\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\".
        primarymusclegroups: Example: \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"biceps,triceps,chest\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\". See possible musclegroups via the /getMuscleGroups endpoint.
        secondarycolor: The color that the highlighted **secondary** musclegroups should have. The value of the parameter should contain the rgb values as a string like \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"red,green,blue\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\".
        transparentbackground: Make the background transparent when set to 1. The default value is 0.
        
    """
    url = f"https://muscle-group-image-generator.p.rapidapi.com/getMulticolorImage"
    querystring = {'secondaryMuscleGroups': secondarymusclegroups, 'primaryColor': primarycolor, 'primaryMuscleGroups': primarymusclegroups, 'secondaryColor': secondarycolor, }
    if transparentbackground:
        querystring['transparentBackground'] = transparentbackground
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "muscle-group-image-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

