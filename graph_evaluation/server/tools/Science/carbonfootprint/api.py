import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def airqualityhealthindex(o3: str, no2: str, pm: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the official air quality health index (1 to 10) bases on key parameters.The national AQHI is based on three-hour average concentrations of ground-level ozone (O3), nitrogen dioxide (NO2), and fine particulate matter (PM2.5). O3 and NO2 are measured in parts per billion (ppb) while PM2.5 is 	 measured in micrograms per cubic metre (ug/m3)"
    o3: The ground-level ozone (O3) in parts per billion (ppb).in 
        no2: The nitrogen dioxide (NO2),  in parts per billion (ppb)
        pm: The fine particulate matter (PM2.5), PM2.5 is 	 * measured in micrograms per cubic metre (ug/m3).
        
    """
    url = f"https://carbonfootprint1.p.rapidapi.com/AirQualityHealthIndex"
    querystring = {'O3': o3, 'NO2': no2, 'PM': pm, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonfootprint1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def treeequivalent(weight: str, unit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate how many trees it took to create paper."
    weight: The weight of the paper
        unit: The unit (kg or lb) used for the weight
        
    """
    url = f"https://carbonfootprint1.p.rapidapi.com/TreeEquivalent"
    querystring = {'weight': weight, 'unit': unit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonfootprint1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def traditionalhydrotocarbonfootprint(consumption: str, location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate CO2e from the use of traditional hydro provider"
    consumption: The KWH usage of hydro.
        location: The country or continent providing the hydro. Can be any of USA, Canada, UK, Europe, Africa, LatinAmerica, MiddleEast, OtherCountry
        
    """
    url = f"https://carbonfootprint1.p.rapidapi.com/TraditionalHydroToCarbonFootprint"
    querystring = {'consumption': consumption, 'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonfootprint1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cleanhydrotocarbonfootprint(energy: str, consumption: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the CO2e in Kg from the consumption of clean hydro energy"
    energy: The source of the clean energy. Can be Solar, Wind, HydroElectric, Biomass, Geothermal, Tidal or OtherCleanEnergy
        consumption: The amount of energy consumed in KWH..
        
    """
    url = f"https://carbonfootprint1.p.rapidapi.com/CleanHydroToCarbonFootprint"
    querystring = {'energy': energy, 'consumption': consumption, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonfootprint1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fueltoco2e(type: str, litres: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Transform liters of Diesel, Petrol or LPG into CO2 Equivalent in Kg."
    type: The type can be Petrol, Diesel, LPG.
        litres: The number of litres to calculate from.
        
    """
    url = f"https://carbonfootprint1.p.rapidapi.com/FuelToCO2e"
    querystring = {'type': type, 'litres': litres, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonfootprint1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def carbonfootprintfromcartravel(distance: str, vehicle: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the CO2e in Kg from a travel by car"
    distance: The distance in KM.
        vehicle: The type of car, either SmallDieselCar, MediumDieselCar, LargeDieselCar, MediumHybridCar, LargeHybridCar, MediumLPGCar, LargeLPGCar, MediumCNGCar, LargeCNGCar, SmallPetrolVan, LargePetrolVan, SmallDielselVan, MediumDielselVan, LargeDielselVan, LPGVan, CNGVan, SmallPetrolCar, MediumPetrolCar, LargePetrolCar, SmallMotorBike, MediumMotorBike, LargeMotorBike
        
    """
    url = f"https://carbonfootprint1.p.rapidapi.com/CarbonFootprintFromCarTravel"
    querystring = {'distance': distance, 'vehicle': vehicle, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonfootprint1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def carbonfootprintfromflight(distance: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate CO2e in Kg from a travel by air."
    distance: The flight distance in KM
        type: The type of flight, any of DomesticFlight, ShortEconomyClassFlight, ShortBusinessClassFlight, LongEconomyClassFlight, LongPremiumClassFlight, LongBusinessClassFlight, LongFirstClassFlight
        
    """
    url = f"https://carbonfootprint1.p.rapidapi.com/CarbonFootprintFromFlight"
    querystring = {'distance': distance, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonfootprint1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def carbonfootprintfrommotorbike(type: str, distance: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the CO2e in Kg from a motorbike travel"
    type: The type of motorbike, can be any of SmallMotorBike, MediumMotorBike, LargeMotorBike
        distance: The distance in KM
        
    """
    url = f"https://carbonfootprint1.p.rapidapi.com/CarbonFootprintFromMotorBike"
    querystring = {'type': type, 'distance': distance, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonfootprint1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def carbonfootprintfrompublictransit(distance: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return CO2e in Kg from the use of public transporation."
    distance: The distance in KM.
        type: The type of transportation, one of: Taxi, ClassicBus, EcoBus, Coach, NationalTrain, LightRail, Subway, FerryOnFoot, FerryInCar
        
    """
    url = f"https://carbonfootprint1.p.rapidapi.com/CarbonFootprintFromPublicTransit"
    querystring = {'distance': distance, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "carbonfootprint1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

