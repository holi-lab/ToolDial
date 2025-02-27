import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def optifineversionlist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Optifine version list."
    
    """
    url = f"https://nitroxenon-minecraft-forge-v1.p.rapidapi.com/optifine/versionlist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nitroxenon-minecraft-forge-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadforgeinstaller(minecraftversion: str, forgeversion: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Redirect to download Forge Installer."
    
    """
    url = f"https://nitroxenon-minecraft-forge-v1.p.rapidapi.com/forge/getforge/{minecraftversion}/{forgeversion}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nitroxenon-minecraft-forge-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def minecraftresources(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Same to official source. Used to get resources file for MC 1.6."
    
    """
    url = f"https://nitroxenon-minecraft-forge-v1.p.rapidapi.com/resources/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nitroxenon-minecraft-forge-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forgeversionlist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method return a forge version list in JSON format."
    
    """
    url = f"https://nitroxenon-minecraft-forge-v1.p.rapidapi.com/forge/versionlist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nitroxenon-minecraft-forge-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def minecraftversionlist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return Minecraft version list in JSON format."
    
    """
    url = f"https://nitroxenon-minecraft-forge-v1.p.rapidapi.com/versions/versions.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nitroxenon-minecraft-forge-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadoptimize(optifineversion: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Redirect to download Optimize."
    
    """
    url = f"https://nitroxenon-minecraft-forge-v1.p.rapidapi.com/optifine/{optifineversion}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nitroxenon-minecraft-forge-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadminecraftjar(minecraftversion: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Redirect to download Minecraft .jar file."
    
    """
    url = f"https://nitroxenon-minecraft-forge-v1.p.rapidapi.com/versions/{minecraftversion}/{minecraftversion}.jar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nitroxenon-minecraft-forge-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forgelegacylist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a JSON format file of http://files.minecraftforge.net/minecraftforge/index_legacy.html ."
    
    """
    url = f"https://nitroxenon-minecraft-forge-v1.p.rapidapi.com/forge/legacylist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nitroxenon-minecraft-forge-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadminecraftjson(minecraftversion: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Redirect to download Minecraft .json file."
    
    """
    url = f"https://nitroxenon-minecraft-forge-v1.p.rapidapi.com/versions/{minecraftversion}/{minecraftversion}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nitroxenon-minecraft-forge-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadminecraftjarchecksum(minecraftversion: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a .sha1 file used to checksum for downloadMinecraftJar method."
    
    """
    url = f"https://nitroxenon-minecraft-forge-v1.p.rapidapi.com/versions/{minecraftversion}/{minecraftversion}.jar.sha1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nitroxenon-minecraft-forge-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

