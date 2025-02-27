import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ascvd(gender: str, treatmentforbloodpressure: bool, systolicbloodpressureinmmhg: int, hdlcholesterolinmmolperlt: int, diabetes: bool, ageinyears: int, totalcholesterolinmmolperlt: int, smoker: bool, race: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://clinicalmarkers.p.rapidapi.com/ascvd"
    querystring = {'Gender': gender, 'TreatmentForBloodPressure': treatmentforbloodpressure, 'SystolicBloodPressureInMmHg': systolicbloodpressureinmmhg, 'HdlCholesterolInMmolPerLt': hdlcholesterolinmmolperlt, 'Diabetes': diabetes, 'AgeInYears': ageinyears, 'TotalCholesterolInMmolPerLt': totalcholesterolinmmolperlt, 'Smoker': smoker, 'Race': race, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clinicalmarkers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bacterialmeningitisscoreforchildren(csfgramstainpositive: str, seizureatorpriortoinitialpresentation: str, csfproteinover80: str, csf_ancover1000cells: str, peripheralbloodancover10000cells: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://clinicalmarkers.p.rapidapi.com/BacterialMeningitisScoreForChildren"
    querystring = {'CSFGramStainPositive': csfgramstainpositive, 'SeizureAtOrPriorToInitialPresentation': seizureatorpriortoinitialpresentation, 'CSFProteinOver80': csfproteinover80, 'CSF_ANCOver1000Cells': csf_ancover1000cells, 'PeripheralBloodANCOver10000Cells': peripheralbloodancover10000cells, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clinicalmarkers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def apgar(respirationscore: str, skincolorscore: str, heartrateinbpm: int, responsetoirritablestimuliscore: str, activityscore: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://clinicalmarkers.p.rapidapi.com/apgar"
    querystring = {'RespirationScore': respirationscore, 'SkinColorScore': skincolorscore, 'HeartRateInBpm': heartrateinbpm, 'ResponseToIrritableStimuliScore': responsetoirritablestimuliscore, 'ActivityScore': activityscore, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clinicalmarkers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def apache2(fio2: int, arterialph: int, meanarterialpressureinmmhg: int, heartrateinbpm: int, respiratoryrateinbpm: int, acuterenalfailure: bool, bodytemperatureindegcelsius: int, aa: int, pao2: int, hematocrit: int, serumsodiuminmeqperlt: int, postoperative: str, serumpotasiuminmeqperlt: int, age: int, whitebloodcellcountinbillionsperlt: int, severeorganinsufficiencyorimmunocompromised: bool, glasgowcomascore: int, serumcreatinineinmicromolsperlt: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://clinicalmarkers.p.rapidapi.com/apache2"
    querystring = {'_Fio2': fio2, 'ArterialPh': arterialph, 'MeanArterialPressureInMmHg': meanarterialpressureinmmhg, 'HeartRateInBpm': heartrateinbpm, 'RespiratoryRateInBpm': respiratoryrateinbpm, 'AcuteRenalFailure': acuterenalfailure, 'BodyTemperatureInDegCelsius': bodytemperatureindegcelsius, '_Aa': aa, '_PaO2': pao2, 'Hematocrit': hematocrit, 'SerumSodiumInMeqPerLt': serumsodiuminmeqperlt, 'PostOperative': postoperative, 'SerumPotasiumInMeqPerLt': serumpotasiuminmeqperlt, 'Age': age, 'WhiteBloodCellCountInBillionsPerLt': whitebloodcellcountinbillionsperlt, 'SevereOrganInsufficiencyOrImmunocompromised': severeorganinsufficiencyorimmunocompromised, 'GlasgowComaScore': glasgowcomascore, 'SerumCreatinineInMicroMolsPerLt': serumcreatinineinmicromolsperlt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clinicalmarkers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def audit(frequencyofmemorylossduetodrinking: str, frequencyofinabilitytoactproperly: str, frequencyofinabilitytostopdrinking: str, frequencyofremorsefordrinking: str, numberofdrinksperdrinkingday: int, drinkinjury: str, sixdrinksfrenquency: str, frequencyofmorningdrinking: str, cutdownsuggestions: str, drinkfrequency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://clinicalmarkers.p.rapidapi.com/audit"
    querystring = {'FrequencyOfMemoryLossDueToDrinking': frequencyofmemorylossduetodrinking, 'FrequencyOfInabilityToActProperly': frequencyofinabilitytoactproperly, 'FrequencyOfInabilityToStopDrinking': frequencyofinabilitytostopdrinking, 'FrequencyOfRemorseForDrinking': frequencyofremorsefordrinking, 'NumberOfDrinksPerDrinkingDay': numberofdrinksperdrinkingday, 'DrinkInjury': drinkinjury, 'SixDrinksFrenquency': sixdrinksfrenquency, 'FrequencyOfMorningDrinking': frequencyofmorningdrinking, 'CutDownSuggestions': cutdownsuggestions, 'DrinkFrequency': drinkfrequency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clinicalmarkers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bmi(heightincentrimetres: int, weightinkilograms: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://clinicalmarkers.p.rapidapi.com/bmi"
    querystring = {'HeightInCentrimetres': heightincentrimetres, 'WeightInKilograms': weightinkilograms, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clinicalmarkers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bsa(weightinkilograms: int, heightincentrimetres: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://clinicalmarkers.p.rapidapi.com/bsa"
    querystring = {'WeightInKilograms': weightinkilograms, 'HeightInCentrimetres': heightincentrimetres, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clinicalmarkers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

