import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dna2mrna(dna: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This end-point converts a DNA sequence into an mRNA sequence."
    dna: The DNA sequence to transform into an mRNA sequence.
        
    """
    url = f"https://dna2protein.p.rapidapi.com/DNA2mRNA"
    querystring = {'dna': dna, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dna2protein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dna2aminoacid(dna: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Transform a DNA sequence into a sequence of Amino Acids"
    dna: The DNA sequence used for the transformation to Amino Acids
        
    """
    url = f"https://dna2protein.p.rapidapi.com/DNA2AminoAcid"
    querystring = {'dna': dna, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dna2protein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mrna2aminoacid(mrna: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Transform an mRNA sequence into a sequence of Amino Acids."
    mrna: the mRNA sequence used to find the Amino Acid sequence.
        
    """
    url = f"https://dna2protein.p.rapidapi.com/mRNA2AminoAcid"
    querystring = {'mRNA': mrna, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dna2protein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mrna2dna(mrna: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint transforms an mRNA sequence to its DNA sequence equivalent."
    mrna: The mRNA sequence as a string of letters.
        
    """
    url = f"https://dna2protein.p.rapidapi.com/mRNA2DNA"
    querystring = {'mRNA': mrna, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dna2protein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

