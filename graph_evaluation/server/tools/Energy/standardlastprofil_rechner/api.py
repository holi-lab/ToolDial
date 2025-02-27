import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve(is_from: int, to: int=1662170993552, consumption: int=2400, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this API endpoint  you can create electrical profiles by scaling the BDEW profiles to your desired annual demand. The electrical profiles are the standard load profiles from BDEW. All profiles have a resolution of 15 minutes. They are based on measurements in the German electricity sector.
		
		Mit diesem API-Endpunkt können Sie elektrische Profile erstellen, indem Sie die BDEW-Profile auf Ihren gewünschten Jahresbedarf skalieren. Die elektrischen Profile sind die Standardlastprofile des BDEW. Alle Profile haben eine Auflösung von 15 Minuten. Sie basieren auf Messungen in der deutschen Stromwirtschaft.
		
		| Typ	| Beschreibung	| Erläuterung |
		|--------|---------------------|-------------|
		| G0	| Gewerbe allgemein	| Gewogener Mittelwert der Profile G1-G6 |
		| G1	| Gewerbe werktags 8–18 Uhr |	z.B. Büros, Arztpraxen, Werkstätten, Verwaltungseinrichtungen |
		| G2	| Gewerbe mit starkem bis überwiegendem Verbrauch in den Abendstunden	z.B. Sportvereine, Fitnessstudios, Abendgaststätten |
		| G3	| Gewerbe durchlaufend	 | z.B. Kühlhäuser, Pumpen, Kläranlagen |
		| G4	| Laden/Friseur |	 
		| G5	| Bäckerei mit Backstube |	 
		| G6	| Wochenendbetrieb	| z.B. Kinos |
		| G7	| Mobilfunksendestation	| durchgängiges Bandlastprofil |
		| L0	| Landwirtschaftsbetriebe allgemein	| Gewogener Mittelwert der Profile L1 und L2 |
		| L1	| Landwirtschaftsbetriebe mit Milchwirtschaft/Nebenerwerbs-Tierzucht |	 
		| L2	| Übrige Landwirtschaftsbetriebe |	 
		| H0 |  Haushalt |	 
		
		
		Verbrauchsprognose für Strom auf Basis der repräsentativen Standardlastprofile in Deutschland :"
    from: Unix Timestamp (milliseconds since 1970-01-01) in UTC to start profile with.
        to: Unix Timestamp (milliseconds since 1970-01-01) in UTC to end profile with. Defaults to \"now\".
        consumption: Yearly consumption in Kilo-Watt-Hours.

Jahresarbeit in Kilo-Watt-Stunden.
        
    """
    url = f"https://standardlastprofil-rechner.p.rapidapi.com/retrieve"
    querystring = {'from': is_from, }
    if to:
        querystring['to'] = to
    if consumption:
        querystring['consumption'] = consumption
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "standardlastprofil-rechner.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

