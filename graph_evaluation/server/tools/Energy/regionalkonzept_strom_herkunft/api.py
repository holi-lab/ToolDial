import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def erzeugung(postleitzahl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpunkt erlaubt eine Liste der Postleitzahlenbereiche zu erhalten, in denen Verbraucher sein dürfen, die  von einer Erzeugungsanlage von angegebener Postleitzahl ihren Strom beziehen sollen.
		
		**Regionaler Herkunftsnachweis**
		Bei Verwendung von Regionalnachweisen dürfen Stromversorger nun in ihrer Stromkennzeichnung ausweisen, dass der von ihnen gelieferte EEG-Strom aus Anlagen in der Region der Verbraucherin oder des Verbrauchers kommt.  [weitere Informationen](https://www.umweltbundesamt.de/dokument/rnr-regionenkonzept-2021/)
		
		**Verwendung**
		`/erzeugung?postleitzahl=10117 `
		
		Liefert eine Liste der Postleitzahlen, an der eine Verbraucher sein darf, damit der Strom als regionaler Ökostrom bezeichnet werden darf, wenn dessen Netzeinspeisung (Erzeuger) in der Postleitzahl `10117` stattfindet.
		
		Datenstand: 2021"
    postleitzahl: Postleitzahl der Erzeugungsanlage, für die eine Liste der Postleitzahlen von Verbrauchergebiete einers regionalen Ökostrombezugs erstellt werden soll.
        
    """
    url = f"https://regionalkonzept-strom-herkunft.p.rapidapi.com/erzeugung"
    querystring = {'postleitzahl': postleitzahl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "regionalkonzept-strom-herkunft.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def strommix(postleitzahl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Resultierender Strommix nach Anwendung der regionalen Ökostromerzeugung im Gebiet einer Postleitzahl."
    
    """
    url = f"https://regionalkonzept-strom-herkunft.p.rapidapi.com/strommix"
    querystring = {'postleitzahl': postleitzahl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "regionalkonzept-strom-herkunft.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def entnahme(postleitzahl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpunkt erlaubt eine Liste der Postleitzahlenbereiche zu erhalten, in denen Erzeugungsanlagen sein dürfen, die  für einen Verbraucher (Entnahme) an gegebener Postleitzahl ist.
		
		**Regionaler Herkunftsnachweis**
		Bei Verwendung von Regionalnachweisen dürfen Stromversorger nun in ihrer Stromkennzeichnung ausweisen, dass der von ihnen gelieferte EEG-Strom aus Anlagen in der Region der Verbraucherin oder des Verbrauchers kommt.  [weitere Informationen](https://www.umweltbundesamt.de/dokument/rnr-regionenkonzept-2021/)
		
		**Verwendung**
		`/entnahme?postleitzahl=10117 `
		
		Liefert eine Liste der Postleitzahlen, an der eine Erzeugungsanlage sein darf, damit der Strom als regionaler Ökostrom bezeichnet werden darf, wenn die Netzentnahme (Verbrauch) in der Postleitzahl `10117` stattfindet.
		
		Datenstand: 2021"
    postleitzahl: Postleitzahl des Verbrauchers (Entnahmestelle)
        
    """
    url = f"https://regionalkonzept-strom-herkunft.p.rapidapi.com/entnahme"
    querystring = {'postleitzahl': postleitzahl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "regionalkonzept-strom-herkunft.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

