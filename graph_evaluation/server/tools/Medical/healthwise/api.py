import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getassociateddiseases(bodypart: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get common diseases for an organ
		Parameter {bodypart}:
		|  1 | brain          |
		|  2 | eyes           |
		|  3 | ears           |
		|  4 | nose           |
		|  5 | lips           |
		|  6 | tongue         |
		|  7 | jaw            |
		|  8 | neck           |
		|  9 | esophagus      |
		| 10 | lung           |
		| 11 | heart          |
		| 12 | right_arm      |
		| 13 | left_arm       |
		| 14 | elbows         |
		| 15 | hands          |
		| 16 | fingers        |
		| 17 | liver          |
		| 18 | spleen         |
		| 19 | stomach        |
		| 20 | pancreas       |
		| 21 | gallbladder    |
		| 22 | kidneys        |
		| 23 | small_intestin |
		| 24 | colon          |
		| 25 | bladder        |
		| 26 | testicles      |
		| 27 | vagina         |
		| 28 | male_genital   |
		| 29 | female_genital |
		| 30 | anus           |
		| 31 | left_leg       |
		| 32 | right_leg      |
		| 33 | knees          |
		| 34 | foot           |
		| 35 | heel           |
		| 36 | toes           |"
    
    """
    url = f"https://healthwise.p.rapidapi.com/body/diseases/{bodypart}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "healthwise.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcommonsymptoms(organ: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get common symptoms associated to an organ.
		Parameter {organ}:
		|  1 | brain          |
		|  2 | eyes           |
		|  3 | ears           |
		|  4 | nose           |
		|  5 | lips           |
		|  6 | tongue         |
		|  7 | jaw            |
		|  8 | neck           |
		|  9 | esophagus      |
		| 10 | lung           |
		| 11 | heart          |
		| 12 | right_arm      |
		| 13 | left_arm       |
		| 14 | elbows         |
		| 15 | hands          |
		| 16 | fingers        |
		| 17 | liver          |
		| 18 | spleen         |
		| 19 | stomach        |
		| 20 | pancreas       |
		| 21 | gallbladder    |
		| 22 | kidneys        |
		| 23 | small_intestin |
		| 24 | colon          |
		| 25 | bladder        |
		| 26 | testicles      |
		| 27 | vagina         |
		| 28 | male_genital   |
		| 29 | female_genital |
		| 30 | anus           |
		| 31 | left_leg       |
		| 32 | right_leg      |
		| 33 | knees          |
		| 34 | foot           |
		| 35 | heel           |
		| 36 | toes           |"
    
    """
    url = f"https://healthwise.p.rapidapi.com/body/symptoms/{organ}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "healthwise.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdiseasesymptoms(cause: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get common disease symptoms.
		Parameter {cause} :
		|   1 | Macular Degeneration                              |
		|   2 | Cataracts                                         |
		|   3 | Glaucoma                                          |
		|   4 | Diabetic Retinopathy                              |
		|   5 | Dry Eyes Syndrome                                 |
		|   6 | Conjunctivitis (Pinkeye)                          |
		|   7 | Retinal Detachment                                |
		|   8 | Uveitis                                           |
		|   9 | Eyestrain                                         |
		|  10 | Night Blindness (Nyctalopia)                      |
		|  11 | Color Blindness                                   |
		|  12 | Eye Floaters                                      |
		|  13 | Nearsightedness (Myopia)                          |
		|  14 | Farsightedness (Hypermetropia)                    |
		|  15 | Astigmatism                                       |
		|  16 | Presbyopia                                        |
		|  17 | Proptosis                                         |
		|  18 | Strabismus (Crossed Eyes)                         |
		|  19 | Macular Edema                                     |
		|  20 | Dementias                                         |
		|  21 | cancer                                            |
		|  22 | epilepsy                                          |
		|  23 | stroke                                            |
		|  24 | Alzheimer                                         |
		|  25 | migraine                                          |
		|  26 | Balance Disorders                                 |
		|  27 | Cholesteatoma                                     |
		|  28 | Ear Infections                                    |
		|  29 | Ear Ringing (tinnitus)                            |
		|  30 | Hearing Loss                                      |
		|  31 | Otitis                                            |
		|  32 | Perforated eardrum                                |
		|  33 | Sinusitis                                         |
		|  34 | Nasal & Sinus Polyps                              |
		|  35 | flu                                               |
		|  36 | Smell and Taste Disorders                         |
		|  37 | Allergy                                           |
		|  38 | Viral Infection                                   |
		|  39 | Fordyce Glands                                    |
		|  40 | Oral Candidiasis                                  |
		|  41 | Oral Leukoplakia                                  |
		|  42 | Cold Sores (Herpetic Gingivostomatitis)           |
		|  43 | Apthous Ulcers                                    |
		|  44 | Oral Lichen Planus                                |
		|  45 | Mouth Cancer                                      |
		|  46 | Down Syndrome                                     |
		|  47 | Tongue Cancer                                     |
		|  48 | Beckwith-Wiedemann Syndrome                       |
		|  49 | An Overactive Thyroid                             |
		|  50 | Leukemia                                          |
		|  51 | Anemia                                            |
		|  52 | Jaw Fracture                                      |
		|  53 | Dislocated Jaw                                    |
		|  54 | Cleft Palate                                      |
		|  55 | bruxism (teeth grinding)                          |
		|  56 | Wisdom Tooth                                      |
		|  57 | Jawbone Cancer                                    |
		|  58 | Neck Fracture (Cervical Vertebrae)                |
		|  59 | Neck Sprain                                       |
		|  60 | Pinched Nerve                                     |
		|  61 | Spondylosis, or Arthritis In The Neck             |
		|  62 | Whiplash                                          |
		|  63 | Infected Tooth                                    |
		|  64 | Cold or Flu                                       |
		|  65 | Mononucleosis (Mono)                              |
		|  66 | Neck Cancer                                       |
		|  67 | Achalasia                                         |
		|  68 | Barrett's Esophagus                               |
		|  69 | Esophageal Cancer                                 |
		|  70 | Gastroesophageal Reflux Disease (GERD)            |
		|  71 | Gastroparesis                                     |
		|  72 | Peptic Ulcer Disease                              |
		|  73 | Swallowing Disorders                              |
		|  74 | Asthma                                            |
		|  75 | Pneumothorax                                      |
		|  76 | Bronchitis                                        |
		|  77 | COPD (Chronic Obstructive Pulmonary Disease)      |
		|  78 | Lung Cancer                                       |
		|  79 | Pneumonia                                         |
		|  80 | Pulmonary embolus                                 |
		|  81 | Coronary Artery Disease ( CAD )                   |
		|  82 | Heart Failure                                     |
		|  83 | Arrhythmia                                        |
		|  84 | Valvular Heart Disease                            |
		|  85 | Aortic Aneurysms                                  |
		|  86 | Heart Infections                                  |
		|  87 | Heart Problems At Birth                           |
		|  88 | Heart Attack                                      |
		|  89 | Tendinitis and Bursitis                           |
		|  90 | Sprains                                           |
		|  91 | Dislocations                                      |
		|  92 | Bone Fractures                                    |
		|  93 | Nerve Problems                                    |
		|  94 | Osteoarthritis                                    |
		|  95 | Tendinitis and Bursitis                           |
		|  96 | Sprains                                           |
		|  97 | Dislocations                                      |
		|  98 | Bone Fractures                                    |
		|  99 | Nerve Problems                                    |
		| 100 | Osteoarthritis                                    |
		| 101 | High Blood Pressure                               |
		| 102 | Heart Problems                                    |
		| 103 | Arthritis                                         |
		| 104 | Cellulitis                                        |
		| 105 | Tumors                                            |
		| 106 | Ulnar Nerve Entrapment                            |
		| 107 | Osteochondritis Dissecans                         |
		| 108 | Tendon Inflammation                               |
		| 109 | Carpal Tunnel Syndrome                            |
		| 110 | Arthritis                                         |
		| 111 | Dupuytren's Disease                               |
		| 112 | Ganglion Cysts                                    |
		| 113 | Carpal Tunnel Syndrome                            |
		| 114 | Arthritis                                         |
		| 115 | Dupuytren's Disease                               |
		| 116 | Ganglion Cysts                                    |
		| 117 | hepatitis A, B, C                                 |
		| 118 | Cirrhosis                                         |
		| 119 | Liver Cancer                                      |
		| 120 | Hemochromatosis                                   |
		| 121 | Wilson Disease                                    |
		| 122 | Fatty Liver Disease                               |
		| 123 | Malaria                                           |
		| 124 | Hodgkin's Disease                                 |
		| 125 | Leukemia                                          |
		| 126 | Heart Failure                                     |
		| 127 | Cirrhosis                                         |
		| 128 | Tumors                                            |
		| 129 | Viral, Bacterial, or Parasitic Infections         |
		| 130 | Rheumatoid Arthritis                              |
		| 131 | Rheumatoid Gastroesophageal Reflux Disease (GERD) |
		| 132 | Peptic Ulcer Disease (PUD)                        |
		| 133 | Stomach Flu                                       |
		| 134 | Gluten Sensitivity and Celiac Disease             |
		| 135 | Inflammatory Bowel Disease (IBD)                  |
		| 136 | Irritable Bowel Syndrome (IBS)                    |
		| 137 | Constipation                                      |
		| 138 | Stomach Cancer                                    |
		| 139 | Pancreas Malfunction                              |
		| 140 | Pancreatitis                                      |
		| 141 | EAR                                               |
		| 142 | Cystic Fibrosis                                   |
		| 143 | Pancreatic Cancer                                 |
		| 144 | EPI                                               |
		| 145 | Diabetes                                          |
		| 146 | Gallstones                                        |
		| 147 | Cholecystitis                                     |
		| 148 | Choledocholithiasis                               |
		| 149 | Acalculous Gallbladder Disease                    |
		| 150 | Biliary Dyskinesia                                |
		| 151 | Sclerosing Cholangitis                            |
		| 152 | Gallbladder Cancer                                |
		| 153 | Gallbladder Polyps                                |
		| 154 | Chronic kidney disease                            |
		| 155 | Kidney stones                                     |
		| 156 | Glomerulonephritis                                |
		| 157 | Polycystic kidney disease                         |
		| 158 | Urinary tract infections                          |
		| 159 | Kidney Failure                                    |
		| 160 | Kidney Cancer                                     |
		| 161 | Celiac Disease                                    |
		| 162 | Crohn's Disease                                   |
		| 163 | Infections                                        |
		| 164 | Intestinal Cancer                                 |
		| 165 | Intestinal Obstruction                            |
		| 166 | Irritable Bowel Syndrome                          |
		| 167 | Ulcers                                            |
		| 168 | Cystitis                                          |
		| 169 | Urinary Incontinence                              |
		| 170 | Interstitial Cystitis                             |
		| 171 | Overactive Bladder                                |
		| 172 | Epididymitis                                      |
		| 173 | Hydrocele                                         |
		| 174 | Testicular torsion                                |
		| 175 | Varicocele                                        |
		| 176 | Orchitis                                          |
		| 177 | Spermatocele                                      |
		| 178 | Testicule Cancer                                  |
		| 179 | Infections                                        |
		| 180 | chlamydia                                         |
		| 181 | gonorrhoea                                        |
		| 182 | trichomoniasis                                    |
		| 183 | Testicule                                         |
		| 184 | Scrotal Problems                                  |
		| 185 | An Inguinal Hernia                                |
		| 186 | Hypospadias                                       |
		| 187 | Undescended Testicles (Cryptorchidism)            |
		| 188 | Human Papillomavirus (HPV)                        |
		| 189 | Infection Of The Cervix (Cervicitis)              |
		| 190 | Sexually Transmitted Infections (STIs)            |
		| 191 | An Object In The Vagina                           |
		| 192 | Cancer                                            |
		| 193 | Hemorrhoids                                       |
		| 194 | Fissures                                          |
		| 195 | Abscesses                                         |
		| 196 | Achilles Tendinitis                               |
		| 197 | Achilles Tendon Rupture                           |
		| 198 | Baker's Cyst                                      |
		| 199 | Bone Cancer                                       |
		| 200 | Bursitis                                          |
		| 201 | Ankylosing Spondylitis                            |
		| 202 | Achilles Tendinitis                               |
		| 203 | Achilles Tendon Rupture                           |
		| 204 | Baker's Cyst                                      |
		| 205 | Bone Cancer                                       |
		| 206 | Bursitis                                          |
		| 207 | Ankylosing Spondylitis                            |
		| 208 | Osteoarthritis                                    |
		| 209 | Rheumatoid arthritis                              |
		| 210 | Gout                                              |
		| 211 | Pseudogout                                        |
		| 212 | Septic arthritis                                  |
		| 213 | Athlete's foot                                    |
		| 214 | Bunions                                           |
		| 215 | Diabetic neuropathy                               |
		| 216 | Plantar fasciitis                                 |
		| 217 | Corns                                             |
		| 218 | Achilles Tendinitis                               |
		| 219 | Bone Tumor                                        |
		| 220 | Bursitis                                          |
		| 221 | Heel Spur                                         |
		| 222 | Osteomyelitis                                     |
		| 223 | Paget's Disease Of Bone                           |
		| 224 | Blisters                                          |
		| 225 | Claw toe                                          |
		| 226 | Mallet or Hammer Toe                              |
		| 227 | Irritable Bowel Syndrome                          |
		| 228 | Constipation                                      |
		| 229 | Hemorrhoids                                       |
		| 230 | Abscess                                           |
		| 231 | Colitis                                           |
		| 232 | Polyps                                            |
		| 233 | Colon Cancer                                      |"
    
    """
    url = f"https://healthwise.p.rapidapi.com/body/disease_symptoms/{cause}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "healthwise.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmainbodyparts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get main body parts,
		+ Head
		+ Upperbody
		+ Lowerbody"
    
    """
    url = f"https://healthwise.p.rapidapi.com/body"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "healthwise.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbodyorgans(part: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get specefic body part's organs.
		Parameter: {part} :
		|  1 | head      |
		|  2 | upperbody |
		|  3 | lowerbody |"
    
    """
    url = f"https://healthwise.p.rapidapi.com/body/organs/{part}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "healthwise.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

