import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchchannel(cate: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "#Filter channels based on there Languages and categories.
		### Keywords for language and category. 
		> Lang
		
		`Hindi` `Marathi` `Punjabi` `Urdu` `Bengali` `English` `Malayalam` `Tamil` `Gujarati` `Odia` `Telugu` `Bhojpuri` `Kannada` `Assamese` `Nepali` `French`
		
		> Cate
		
		`Entertainment` `Movies` `Kids` `Sports` `Lifestyle` `Infotainment` `Religious` `News` `Music` `Regional` `Devotional` `Business News` `Educational` `Shopping` `Jio Darshan`"
    
    """
    url = f"https://indian-tv-schedule.p.rapidapi.com/searchChannel"
    querystring = {}
    if cate:
        querystring['cate'] = cate
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-tv-schedule.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcategories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Json response of languages list and channel categorie list."
    
    """
    url = f"https://indian-tv-schedule.p.rapidapi.com/getCategories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-tv-schedule.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedule(channel: str, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "#Get todays schedule for a channel.
		## keywords for language.
		#### Press left three dots for more details
		###Hindi
		>`Colors HD` `Sony HD` `Zee TV HD` `Sony Max HD` `Zee Cinema HD` `Aaj Tak` `ABP News India` `MTV HD Plus` `MTV Beats HD` `Mastiii` `Nick Hindi` `Pogo Hindi` `Cartoon Network Hindi` `Sony BBC Earth HD` `Food Food` `Zee Business` `CNBC Awaaz` `Aastha` `Aastha Bhajan` `Sanskar` `Sony SAB` `And TV HD` `Rishtey` `Zee Anmol` `Sony Pal` `Sony SAB HD` `Epic` `Investigation Discovery` `DD India` `DD National` `Dabangg` `DD Madhya Pradesh` `DD Rajasthan (Jaipur)` `Colors Cineplex` `Sony MAX2` `And Pictures HD` `Zee Cinema` `B4U Movies` `JioCinema` `Zee Bollywood` `Enterr 10` `Zee Action` `Zee Anmol Cinema` `News 18 India` `India TV` `NDTV India` `Zee News` `News 24` `Ten3 HD` `Zing` `9XM` `E 24` `B4U Music` `ZOOM` `MTV` `9X Jalwa` `Zee etc` `Sonic Hindi` `Sony Yay Hindi` `Discovery Kids 2` `Nick Junior` `Animal Planet Hindi` `Discovery Channel Hindi` `History TV18 HD Hindi` `Jio Events HD` `Travel XP HD Hindi` `TLC Hindi` `Peace of Mind` `Divya TV` `Paras tv` `Sadhna` `Satsang TV` `Jinvani TV` `Shubh TV` `Sai Leela` `Arihant TV` `Raj Pariwar` `DD Bihar` `DD Uttar Pradesh` `Arre HD` `Andy Haryana` `Dillagi TV` `Dhamaal TV` `Maha Movies` `Housefull Movies` `Jio Bollywood Premium HD` `News Nation` `India news` `Tez` `News18 RAJASTHAN` `News State UK UP` `Delhi Aaj Tak` `News18 MP` `Kashish News` `DD Sports` `Music India` `PEACE MUSIC` `Zee News MP Chattisgarh` `IBC-24` `Zee Rajasthan` `News18 UP` `DD News` `India News UP` `DD Kisan` `Boogle Bollywood` `AstroFlix` `Jyotish TV` `Disha tv` `Vedic TV` `Darshan 24` `Bhaktisagar 2` `Ishwar TV` `Hare krsna` `Swar Shree` `Shubhsandesh TV` `Katyayani` `Sarv Dharam Sangam` `Lord Buddha` `Om Shanti` `Soham TV` `Awakening` `SRMD HD` `Hare Krsna Music` `Jan TV` `India News Haryana` `India News Rajasthan` `Sahara Samay Bihar` `Zee Hindustan` `Hindi Khabar` `First India News` `News11` `SAHARA SAMAY MP` `India News MP` `Zee UP UK` `Sudarshan` `JK 24x7 News` `Taaza TV` `Live Today` `Sahara Samay Rastriya` `Mh One News` `Janta TV` `News18 Haryana and HP News` `Total TV` `SAHARA SAMAY UP` `Bharat Samachar` `K News India` `Samay Rajasthan` `Loksabha TV` `DD Rajyasabha` `News 1 India` `APN News` `TV 100` `Bansal News` `India Voice` `A1 TV Rajasthan` `INH 24x7` `News India 24x7` `Living India News` `SMBC TV` `Sadhna News Plus` `News World India` `Prime News` `Prudent` `Khabar Fast` `Channel One News` `HNN 24x7` `DNN` `Jantantra` `DD bharati` `Care World` `Star Plus HD IP` `Star Utsav HD` `Star Bharat HD` `Star Gold HD` `Movies OK HD` `Star Sports Hindi 1` `Home Shop 18` `ABP Ganga` `Hare Krsna Pravachan` `JioCinema Shows` `JioCinema Action` `JioCinema Comedy` `Rajyoga TV` `Mandir Shri Govinddevji-Jaipur` `Nimbark TV` `Click Life` `TV9 Bharatvarsh` `Surya Samachar` `Surya Cinema` `Surya Bhakti` `FM News` `Total Tv Haryana` `Bflix Movies` `iLove` `ABZY Dhakad` `ABZY Cool` `ABZY Movies` `Box Cinema` `Prarthana Bhawan` `Manoranjan Grand` `Manoranjan TV` `Mahavir Mandir Patna` `Insync` `ANB News` `Shri Omkareshwar Mandir` `ISKCON Darshan ` `Stars Tell` `Wellness` `B4U Kadak` `ISKCON Darshan  2` `Jio Cricket Hindi HD` 
		
		###Tamil
		>`Polimer News` `Zee Tamil` `Colors Tamil HD` `Sun TV HD` `KTV HD` `K TV` `Thanthi TV` `Sun Music HD` `Jaya Max` `Nick Tamil` `Pogo Tamil` `Cartoon Network Tamil` `D Tamil` `History TV18 HD Tamil` `Travel XP Tamil` `Sony BBC Earth HD Tamil` `Sai TV` `Angel TV HD` `Jaya TV HD` `MK TV` `Vendhar TV` `Sun TV` `Peppers TV` `Puthu Yugam` `Polimer TV` `Raj TV` `Makkal TV` `Adithya TV` `Captain tv` `DD5 Podhigai` `Vasanth TV` `J Movies` `Sun Life` `Raj Digital Plus` `Puthiya Thalimurai` `News7 Tamil` `Sun News` `News 18 Tamilnadu` `MK Music` `Tunes 6` `Raj Musix` `Sahana Music` `Jaya Plus` `Malai Murasu` `Chutti TV` `sonic Tamil` `Discovery Kids Tamil` `Nambikkai` `Captain News` `Sathiyam TV` `Lotus News` `Madhimugam TV` `Win TV` `Raj News 24x7` `Sony Yay Tamil` `CN HD  Tamil` `Star Vijay HD` `Kalaignar TV` `Kalaignar Seithigal ` `Animal Planet HD Tamil` `Aastha Tamil` 
		
		###Telugu
		>`TV9 Telugu News` `Zee Telugu` `Zee Cinemalu` `Gemini Movies HD` `Gemini Movies` `ETV Telugu` `Sakshi tv` `NTV` `Gemini Music HD` `Gemini Music` `Nick Telugu` `Discovery Channel Telugu` `History TV18 HD Telugu` `Sony BBC Earth HD Telugu` `ETV Abhiruchi` `Bhakti TV` `Sri Venkateshwar Bhakti` `CVR OM Spiritual` `Aradhana TV` `Hindu Dharmam` `Subhavartha TV` `Divya Vani` `Calvary` `Gemini TV HD` `ETV Plus` `Gemini TV` `Gemini Comedy` `DD Saptagiri` `Vissa TV` `ETV Cinema` `Gemini Life` `ABN Andhra Jyothi` `V6 News` `ETV Andhra pradesh` `Raj Music Telugu` `10 TV` `TV 5 News` `CVR News` `T News` `ETV Telangana` `Maha News` `HM TV` `Kushi TV` `Cartoon Network Telugu` `ETV Life` `Vanitha` `CVR Health` `TV1` `I News` `Mojo TV` `Raj News Telugu` `Sony Yay Telugu` `CN HD  Telugu` `Maa TV HD` `Maa Gold HD` `Maa Movies HD` `Sonic Telugu` `Aastha Telugu` `Studio One` `Nireekshana TV` 
		
		###Marathi
		>`BBC Marathi` `ABP Majha` `Zee Marathi` `Colors Marathi HD` `Zee Yuva` `TV9 Maharashtra` `Zee 24 Taas` `Saam Tv` `Sony Marathi SD` `DD Sahayadri` `Maiboli` `Rangamanch` `Zee Talkies` `Fakt Marathi` `IBN Lokmat` `Sangeet Marathi` `9x Jhakaas` `In Goa 24x7` `Shree Siddhivinayak` `Tulja Bhavani` `MKN` `Star Pravah HD` `Nick Marathi` `Sonic Marathi` `Maharashtra1` `Lord Shri Vitthal Rukmini` `Dagdusheth Ganpati - Pune` `Ambabai Temple-Kolhapur` `Sugran` `Lokshahi News` 
		
		###Kannada
		>`TV9 Karnataka` `Colors Kannada HD` `Zee Kannada` `Udaya Movies` `Public TV` `BTV` `Udaya Music` `Public Music` `Samara News` `Ayush TV` `Sri Sankara` `Colors Super` `Udaya HD` `Udaya TV` `DD9 chandana (kannada)` `Kasturi` `Udaya Comedy` `Namma TV` `Dighvijay TV` `News18 Kannada News` `TV 5 Kannada` `Raj Music Kannada` `Prajaa TV` `Chintu TV` `Raj News Kannada` `Suvarna HD` `Suvarna Plus HD` `Sonic Kannada` `Nick Kannada` `Aastha Kannada` `Sh Krishna Matta Adamaru Paryaya | 2020-2022, UDUPI` 
		
		###Bengali
		>`ABP Ananda` `Zee Bangla` `Colors Bengali HD` `24 Ghanta TV` `Sangeet Bangla` `Discovery Channel Bengali` `DD Bangla` `Sony aath` `Aakash Aath` `Zee Bangla Cinema` `Amaar Cinema` `News18 Bangla News` `News Time TV` `Bangla Time` `R Plus` `Onkar Only Truth` `Sadhna  News` `Star Jalsha HD` `Jalsha Movies HD` `Nick Bangla` `Sonic Bangla` `Headlines Tripura` `Kolkata Live` `Ctvn Akd Plus` `Calcutta News` `Kolkata TV` `Khushboo TV` `Orange TV` `TV Bangla` 
		
		###Punjabi
		>`PTC Punjabi` `PTC Music` `Desi Channel` `mh1 Shraddha` `PTC Simran` `Sanjha TV` `Garv Punjabi` `JUSPunjabi` `DD Punjabi` `Pitaara` `PTC Punjabi Gold ` `PTC Chak De` `9X Tashan` `PTC DHOL TV` `mh1 (Music)` `Only Music` `PBN Music` `Steelbird Music` `Garv Punjabi Gurbani` `Fateh TV` `JUSOne` `Zee Punjabi HP Haryana` `Chardikla Time TV` `PTC News` `JUS 24x7` `PTC VR` `Global Sanjh` `Manoranjan Movies` `ABP Sanjha` `Takht Sri Patna Sahib, Patna` 
		
		###Malayalam
		>`Surya TV` `Kaumudy TV` `Mazavali Manorama HD` `Surya HD` `Flower TV` `Mazhavil Manorama` `DD Malayalam` `Kairali TV` `Amrita TV` `Kairali News` `Kairali WE TV` `Kiran TV` `Manorama News` `Mathrubhumi News` `Surya Music` `Kappa TV` `Raj Music Malayalam` `Media One TV` `Janam TV` `News 18 Kerala` `Propex TV` `testii` `Shalom` `Jaihind tv` `Jeevan TV` `Raj News Malayalam` `Kochu TV` `Asianet HD` `Asianet Plus HD` `Asianet Movies HD` `Nick Malayalam` `Sonic Malayalam` `Rajyoga Malayalam` `Twenty Four News` `Vignesh TV` `Reporter TV` 
		
		###Gujarati
		>`Colors Gujarati` `Lakshya TV` `DD Girnar` `News18 Gujarati` `Tv 9 Gujarat` `ABP Asmita` `VTV Gujarati` `Sandesh News` `Zee 24 Kalak` `CNBC Bazaar (MNO)` `Valambhakti TV` `Kartavya TV` `Hamari Sanskruti` `Vande Gujarat 1` `Vande Gujarat 10` `Vande Gujarat 11` `Vande Gujarat 12` `Vande Gujarat 13` `Vande Gujarat 14` `Vande Gujarat 15` `Vande Gujarat 16` `Vande Gujarat 2` `Vande Gujarat 3` `Vande Gujarat 4` `Vande Gujarat 5` `Vande Gujarat 6` `Vande Gujarat 7` `Vande Gujarat 8` `Vande Gujarat 9` `GS TV` `Mantavya News` `Nick Gujarati` `Sonic Gujarati` `Digishala` 
		
		###Odia
		>`Prathana TV` `Tarang Music` `Sarthak TV` `Colors Oriya` `DD Oriya` `Tarang TV` `OTV (Odisha TV)` `Alankar TV` `Kanak News` `Zee Kalinga` `Prameya News 7` `News18 Oriya` `Kalinga TV` `MBC` `Naxatra News` `Manjari TV` 
		
		###Bhojpuri
		>`Bhojpuri Cinema` `Sangeet Bhojpuri` `Anjan TV` `Dangal` `Dishum TV` `Oscar Movies` `News18 BIHAR` `Zee Purvaiya` `B4U Bhojpuri` 
		
		###Assamese
		>`Newslive` `Jonack` `Rang` `Rengoni` `DD13 Guwahati NE` `Indradhanu` `DY 365` `Prime Time News` `News 18 Assam` `Prag News` `Ramdhenu` `Assam Talks` `North East Live` 
		
		###English
		>`Jio Cricket English HD` `HBO HD` `Sony Six HD` `Times NOW` `Ten 1HD` `Ten2 HD` `CNN NEWS 18` `Republic TV` `Discovery HD World` `History TV18 HD` `TLC HD` `GOOD TiMES` `CNBC Tv18 Prime HD` `NDTV Profit` `Millionlights` `Swayam Prabha 01 VAGEESH` `Swayam Prabha 02 SANSKRITI` `Swayam Prabha 03 PRABODH` `Jio Exclusive HD` `Comedy Central HD` `Colors Infinity HD` `AXN HD` `Sony Pix HD` `WB` `Movies Now HD` `Times Now World` `NDTV 24x7` `Eurosport HD` `Jio Football HD` `Jio Cricket 1 HD` `Jio Football 1 HD` `Jio Football 2 HD` `Ten 1` `Ten 2` `Ten 3` `Sony Six SD` `Nickelodeon` `CN HD  English` `FYI TV18 HD` `Animal Planet HD World` `Travel XP HD` `Discovery Science` `Animal Planet English` `Discovery` `TLC English` `Sadhguru Television HD` `CNBC Tv 18` `ET Now` `Swayam Prabha 04 SAARASWAT` `Swayam Prabha 05 PRABANDHAN` `Swayam Prabha 06 VIDHIK` `Swayam Prabha 07 KAUTILYA` `Swayam Prabha 08 ARYABHATT` `Swayam Prabha 09 SPANDAN` `Swayam Prabha 10 DAKSH` `Swayam Prabha 11 NPTEL Chemical Engg` `Swayam Prabha 12 NPTEL Civil Engg` `Swayam Prabha 13 NPTEL Computer Engg` `Swayam Prabha 14 NPTEL Electrical Engg` `Swayam Prabha 15 NPTEL Engg Science` `Swayam Prabha 16 NPTEL Humanities` `Swayam Prabha 17 NPTEL Mech Engg` `Swayam Prabha 18 NPTEL Mathematics` `Swayam Prabha 19 IIT PAL Biology` `Swayam Prabha 20 IIT PAL Chemistry` `Swayam Prabha 21 IIT PAL Mathematics` `Swayam Prabha 22 IIT PAL Physics` `Swayam Prabha 23 IGNOU Liberal Arts` `Swayam Prabha 24 IGNOU Agriculture` `Swayam Prabha 25 GYAN DARSHAN` `Swayam Prabha 26 IGNOU Open Univ` `Swayam Prabha 27 NIOS S S Edu` `Swayam Prabha 28 NIOS H S S Edu` `Swayam Prabha 29 UGC INFLIBNET` `The Q India` `MNX HD` `MN  HD` `MNX` `Romedy Now HD` `India Today` `Brit Asia` `Hi Dost!` `BBC World News` `Discovery Turbo` `Sony BBC Earth HD English` `Swayam Prabha 30 NIOS Gyanamrit` `Swayam Prabha 31 NCERT` `Swayam Prabha 32 IGNOU and NIOS` `Testton` `News X` `Mirror Now` `CNN` `AL Jazeera` `Wion` `dw` `TV5 Monde` `News 9` `GoodNews Channel` `Channel News Asia International` `Euro News` `RDX Goa` `CVR English` `Nickelodeon Jr.` `Insight` `Star Sports Select HD1` `Star Sports 1` `Star Sports 2` `Star Sports Select HD2` `Jio Cricket 2 HD` `Jio Cricket 3 HD` `Highbrow` `Nick HD ` `God TV` `Eurosport` `HHDL` `Top Tutor` `RT TV` 
		
		###Nepali
		>`Nepal one` 
		
		###French
		>`France 24` 
		
		###Urdu
		>`Mercy TV` `DD urdu` `DD Kashir` `Zee Salaam` `Channel Win` `Gulistan News` `News18 Urdu` `4 TV` `Tehzeeb TV`"
    
    """
    url = f"https://indian-tv-schedule.p.rapidapi.com/Schedule"
    querystring = {'channel': channel, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-tv-schedule.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def todayschedule(channel: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "#Get todays schedule for a channel.
		## keywords for language.
		#### Press left three dots for more details
		###Hindi
		>`Colors HD` `Sony HD` `Zee TV HD` `Sony Max HD` `Zee Cinema HD` `Aaj Tak` `ABP News India` `MTV HD Plus` `MTV Beats HD` `Mastiii` `Nick Hindi` `Pogo Hindi` `Cartoon Network Hindi` `Sony BBC Earth HD` `Food Food` `Zee Business` `CNBC Awaaz` `Aastha` `Aastha Bhajan` `Sanskar` `Sony SAB` `And TV HD` `Rishtey` `Zee Anmol` `Sony Pal` `Sony SAB HD` `Epic` `Investigation Discovery` `DD India` `DD National` `Dabangg` `DD Madhya Pradesh` `DD Rajasthan (Jaipur)` `Colors Cineplex` `Sony MAX2` `And Pictures HD` `Zee Cinema` `B4U Movies` `JioCinema` `Zee Bollywood` `Enterr 10` `Zee Action` `Zee Anmol Cinema` `News 18 India` `India TV` `NDTV India` `Zee News` `News 24` `Ten3 HD` `Zing` `9XM` `E 24` `B4U Music` `ZOOM` `MTV` `9X Jalwa` `Zee etc` `Sonic Hindi` `Sony Yay Hindi` `Discovery Kids 2` `Nick Junior` `Animal Planet Hindi` `Discovery Channel Hindi` `History TV18 HD Hindi` `Jio Events HD` `Travel XP HD Hindi` `TLC Hindi` `Peace of Mind` `Divya TV` `Paras tv` `Sadhna` `Satsang TV` `Jinvani TV` `Shubh TV` `Sai Leela` `Arihant TV` `Raj Pariwar` `DD Bihar` `DD Uttar Pradesh` `Arre HD` `Andy Haryana` `Dillagi TV` `Dhamaal TV` `Maha Movies` `Housefull Movies` `Jio Bollywood Premium HD` `News Nation` `India news` `Tez` `News18 RAJASTHAN` `News State UK UP` `Delhi Aaj Tak` `News18 MP` `Kashish News` `DD Sports` `Music India` `PEACE MUSIC` `Zee News MP Chattisgarh` `IBC-24` `Zee Rajasthan` `News18 UP` `DD News` `India News UP` `DD Kisan` `Boogle Bollywood` `AstroFlix` `Jyotish TV` `Disha tv` `Vedic TV` `Darshan 24` `Bhaktisagar 2` `Ishwar TV` `Hare krsna` `Swar Shree` `Shubhsandesh TV` `Katyayani` `Sarv Dharam Sangam` `Lord Buddha` `Om Shanti` `Soham TV` `Awakening` `SRMD HD` `Hare Krsna Music` `Jan TV` `India News Haryana` `India News Rajasthan` `Sahara Samay Bihar` `Zee Hindustan` `Hindi Khabar` `First India News` `News11` `SAHARA SAMAY MP` `India News MP` `Zee UP UK` `Sudarshan` `JK 24x7 News` `Taaza TV` `Live Today` `Sahara Samay Rastriya` `Mh One News` `Janta TV` `News18 Haryana and HP News` `Total TV` `SAHARA SAMAY UP` `Bharat Samachar` `K News India` `Samay Rajasthan` `Loksabha TV` `DD Rajyasabha` `News 1 India` `APN News` `TV 100` `Bansal News` `India Voice` `A1 TV Rajasthan` `INH 24x7` `News India 24x7` `Living India News` `SMBC TV` `Sadhna News Plus` `News World India` `Prime News` `Prudent` `Khabar Fast` `Channel One News` `HNN 24x7` `DNN` `Jantantra` `DD bharati` `Care World` `Star Plus HD IP` `Star Utsav HD` `Star Bharat HD` `Star Gold HD` `Movies OK HD` `Star Sports Hindi 1` `Home Shop 18` `ABP Ganga` `Hare Krsna Pravachan` `JioCinema Shows` `JioCinema Action` `JioCinema Comedy` `Rajyoga TV` `Mandir Shri Govinddevji-Jaipur` `Nimbark TV` `Click Life` `TV9 Bharatvarsh` `Surya Samachar` `Surya Cinema` `Surya Bhakti` `FM News` `Total Tv Haryana` `Bflix Movies` `iLove` `ABZY Dhakad` `ABZY Cool` `ABZY Movies` `Box Cinema` `Prarthana Bhawan` `Manoranjan Grand` `Manoranjan TV` `Mahavir Mandir Patna` `Insync` `ANB News` `Shri Omkareshwar Mandir` `ISKCON Darshan ` `Stars Tell` `Wellness` `B4U Kadak` `ISKCON Darshan  2` `Jio Cricket Hindi HD` 
		
		###Tamil
		>`Polimer News` `Zee Tamil` `Colors Tamil HD` `Sun TV HD` `KTV HD` `K TV` `Thanthi TV` `Sun Music HD` `Jaya Max` `Nick Tamil` `Pogo Tamil` `Cartoon Network Tamil` `D Tamil` `History TV18 HD Tamil` `Travel XP Tamil` `Sony BBC Earth HD Tamil` `Sai TV` `Angel TV HD` `Jaya TV HD` `MK TV` `Vendhar TV` `Sun TV` `Peppers TV` `Puthu Yugam` `Polimer TV` `Raj TV` `Makkal TV` `Adithya TV` `Captain tv` `DD5 Podhigai` `Vasanth TV` `J Movies` `Sun Life` `Raj Digital Plus` `Puthiya Thalimurai` `News7 Tamil` `Sun News` `News 18 Tamilnadu` `MK Music` `Tunes 6` `Raj Musix` `Sahana Music` `Jaya Plus` `Malai Murasu` `Chutti TV` `sonic Tamil` `Discovery Kids Tamil` `Nambikkai` `Captain News` `Sathiyam TV` `Lotus News` `Madhimugam TV` `Win TV` `Raj News 24x7` `Sony Yay Tamil` `CN HD  Tamil` `Star Vijay HD` `Kalaignar TV` `Kalaignar Seithigal ` `Animal Planet HD Tamil` `Aastha Tamil` 
		
		###Telugu
		>`TV9 Telugu News` `Zee Telugu` `Zee Cinemalu` `Gemini Movies HD` `Gemini Movies` `ETV Telugu` `Sakshi tv` `NTV` `Gemini Music HD` `Gemini Music` `Nick Telugu` `Discovery Channel Telugu` `History TV18 HD Telugu` `Sony BBC Earth HD Telugu` `ETV Abhiruchi` `Bhakti TV` `Sri Venkateshwar Bhakti` `CVR OM Spiritual` `Aradhana TV` `Hindu Dharmam` `Subhavartha TV` `Divya Vani` `Calvary` `Gemini TV HD` `ETV Plus` `Gemini TV` `Gemini Comedy` `DD Saptagiri` `Vissa TV` `ETV Cinema` `Gemini Life` `ABN Andhra Jyothi` `V6 News` `ETV Andhra pradesh` `Raj Music Telugu` `10 TV` `TV 5 News` `CVR News` `T News` `ETV Telangana` `Maha News` `HM TV` `Kushi TV` `Cartoon Network Telugu` `ETV Life` `Vanitha` `CVR Health` `TV1` `I News` `Mojo TV` `Raj News Telugu` `Sony Yay Telugu` `CN HD  Telugu` `Maa TV HD` `Maa Gold HD` `Maa Movies HD` `Sonic Telugu` `Aastha Telugu` `Studio One` `Nireekshana TV` 
		
		###Marathi
		>`BBC Marathi` `ABP Majha` `Zee Marathi` `Colors Marathi HD` `Zee Yuva` `TV9 Maharashtra` `Zee 24 Taas` `Saam Tv` `Sony Marathi SD` `DD Sahayadri` `Maiboli` `Rangamanch` `Zee Talkies` `Fakt Marathi` `IBN Lokmat` `Sangeet Marathi` `9x Jhakaas` `In Goa 24x7` `Shree Siddhivinayak` `Tulja Bhavani` `MKN` `Star Pravah HD` `Nick Marathi` `Sonic Marathi` `Maharashtra1` `Lord Shri Vitthal Rukmini` `Dagdusheth Ganpati - Pune` `Ambabai Temple-Kolhapur` `Sugran` `Lokshahi News` 
		
		###Kannada
		>`TV9 Karnataka` `Colors Kannada HD` `Zee Kannada` `Udaya Movies` `Public TV` `BTV` `Udaya Music` `Public Music` `Samara News` `Ayush TV` `Sri Sankara` `Colors Super` `Udaya HD` `Udaya TV` `DD9 chandana (kannada)` `Kasturi` `Udaya Comedy` `Namma TV` `Dighvijay TV` `News18 Kannada News` `TV 5 Kannada` `Raj Music Kannada` `Prajaa TV` `Chintu TV` `Raj News Kannada` `Suvarna HD` `Suvarna Plus HD` `Sonic Kannada` `Nick Kannada` `Aastha Kannada` `Sh Krishna Matta Adamaru Paryaya | 2020-2022, UDUPI` 
		
		###Bengali
		>`ABP Ananda` `Zee Bangla` `Colors Bengali HD` `24 Ghanta TV` `Sangeet Bangla` `Discovery Channel Bengali` `DD Bangla` `Sony aath` `Aakash Aath` `Zee Bangla Cinema` `Amaar Cinema` `News18 Bangla News` `News Time TV` `Bangla Time` `R Plus` `Onkar Only Truth` `Sadhna  News` `Star Jalsha HD` `Jalsha Movies HD` `Nick Bangla` `Sonic Bangla` `Headlines Tripura` `Kolkata Live` `Ctvn Akd Plus` `Calcutta News` `Kolkata TV` `Khushboo TV` `Orange TV` `TV Bangla` 
		
		###Punjabi
		>`PTC Punjabi` `PTC Music` `Desi Channel` `mh1 Shraddha` `PTC Simran` `Sanjha TV` `Garv Punjabi` `JUSPunjabi` `DD Punjabi` `Pitaara` `PTC Punjabi Gold ` `PTC Chak De` `9X Tashan` `PTC DHOL TV` `mh1 (Music)` `Only Music` `PBN Music` `Steelbird Music` `Garv Punjabi Gurbani` `Fateh TV` `JUSOne` `Zee Punjabi HP Haryana` `Chardikla Time TV` `PTC News` `JUS 24x7` `PTC VR` `Global Sanjh` `Manoranjan Movies` `ABP Sanjha` `Takht Sri Patna Sahib, Patna` 
		
		###Malayalam
		>`Surya TV` `Kaumudy TV` `Mazavali Manorama HD` `Surya HD` `Flower TV` `Mazhavil Manorama` `DD Malayalam` `Kairali TV` `Amrita TV` `Kairali News` `Kairali WE TV` `Kiran TV` `Manorama News` `Mathrubhumi News` `Surya Music` `Kappa TV` `Raj Music Malayalam` `Media One TV` `Janam TV` `News 18 Kerala` `Propex TV` `testii` `Shalom` `Jaihind tv` `Jeevan TV` `Raj News Malayalam` `Kochu TV` `Asianet HD` `Asianet Plus HD` `Asianet Movies HD` `Nick Malayalam` `Sonic Malayalam` `Rajyoga Malayalam` `Twenty Four News` `Vignesh TV` `Reporter TV` 
		
		###Gujarati
		>`Colors Gujarati` `Lakshya TV` `DD Girnar` `News18 Gujarati` `Tv 9 Gujarat` `ABP Asmita` `VTV Gujarati` `Sandesh News` `Zee 24 Kalak` `CNBC Bazaar (MNO)` `Valambhakti TV` `Kartavya TV` `Hamari Sanskruti` `Vande Gujarat 1` `Vande Gujarat 10` `Vande Gujarat 11` `Vande Gujarat 12` `Vande Gujarat 13` `Vande Gujarat 14` `Vande Gujarat 15` `Vande Gujarat 16` `Vande Gujarat 2` `Vande Gujarat 3` `Vande Gujarat 4` `Vande Gujarat 5` `Vande Gujarat 6` `Vande Gujarat 7` `Vande Gujarat 8` `Vande Gujarat 9` `GS TV` `Mantavya News` `Nick Gujarati` `Sonic Gujarati` `Digishala` 
		
		###Odia
		>`Prathana TV` `Tarang Music` `Sarthak TV` `Colors Oriya` `DD Oriya` `Tarang TV` `OTV (Odisha TV)` `Alankar TV` `Kanak News` `Zee Kalinga` `Prameya News 7` `News18 Oriya` `Kalinga TV` `MBC` `Naxatra News` `Manjari TV` 
		
		###Bhojpuri
		>`Bhojpuri Cinema` `Sangeet Bhojpuri` `Anjan TV` `Dangal` `Dishum TV` `Oscar Movies` `News18 BIHAR` `Zee Purvaiya` `B4U Bhojpuri` 
		
		###Assamese
		>`Newslive` `Jonack` `Rang` `Rengoni` `DD13 Guwahati NE` `Indradhanu` `DY 365` `Prime Time News` `News 18 Assam` `Prag News` `Ramdhenu` `Assam Talks` `North East Live` 
		
		###English
		>`Jio Cricket English HD` `HBO HD` `Sony Six HD` `Times NOW` `Ten 1HD` `Ten2 HD` `CNN NEWS 18` `Republic TV` `Discovery HD World` `History TV18 HD` `TLC HD` `GOOD TiMES` `CNBC Tv18 Prime HD` `NDTV Profit` `Millionlights` `Swayam Prabha 01 VAGEESH` `Swayam Prabha 02 SANSKRITI` `Swayam Prabha 03 PRABODH` `Jio Exclusive HD` `Comedy Central HD` `Colors Infinity HD` `AXN HD` `Sony Pix HD` `WB` `Movies Now HD` `Times Now World` `NDTV 24x7` `Eurosport HD` `Jio Football HD` `Jio Cricket 1 HD` `Jio Football 1 HD` `Jio Football 2 HD` `Ten 1` `Ten 2` `Ten 3` `Sony Six SD` `Nickelodeon` `CN HD  English` `FYI TV18 HD` `Animal Planet HD World` `Travel XP HD` `Discovery Science` `Animal Planet English` `Discovery` `TLC English` `Sadhguru Television HD` `CNBC Tv 18` `ET Now` `Swayam Prabha 04 SAARASWAT` `Swayam Prabha 05 PRABANDHAN` `Swayam Prabha 06 VIDHIK` `Swayam Prabha 07 KAUTILYA` `Swayam Prabha 08 ARYABHATT` `Swayam Prabha 09 SPANDAN` `Swayam Prabha 10 DAKSH` `Swayam Prabha 11 NPTEL Chemical Engg` `Swayam Prabha 12 NPTEL Civil Engg` `Swayam Prabha 13 NPTEL Computer Engg` `Swayam Prabha 14 NPTEL Electrical Engg` `Swayam Prabha 15 NPTEL Engg Science` `Swayam Prabha 16 NPTEL Humanities` `Swayam Prabha 17 NPTEL Mech Engg` `Swayam Prabha 18 NPTEL Mathematics` `Swayam Prabha 19 IIT PAL Biology` `Swayam Prabha 20 IIT PAL Chemistry` `Swayam Prabha 21 IIT PAL Mathematics` `Swayam Prabha 22 IIT PAL Physics` `Swayam Prabha 23 IGNOU Liberal Arts` `Swayam Prabha 24 IGNOU Agriculture` `Swayam Prabha 25 GYAN DARSHAN` `Swayam Prabha 26 IGNOU Open Univ` `Swayam Prabha 27 NIOS S S Edu` `Swayam Prabha 28 NIOS H S S Edu` `Swayam Prabha 29 UGC INFLIBNET` `The Q India` `MNX HD` `MN  HD` `MNX` `Romedy Now HD` `India Today` `Brit Asia` `Hi Dost!` `BBC World News` `Discovery Turbo` `Sony BBC Earth HD English` `Swayam Prabha 30 NIOS Gyanamrit` `Swayam Prabha 31 NCERT` `Swayam Prabha 32 IGNOU and NIOS` `Testton` `News X` `Mirror Now` `CNN` `AL Jazeera` `Wion` `dw` `TV5 Monde` `News 9` `GoodNews Channel` `Channel News Asia International` `Euro News` `RDX Goa` `CVR English` `Nickelodeon Jr.` `Insight` `Star Sports Select HD1` `Star Sports 1` `Star Sports 2` `Star Sports Select HD2` `Jio Cricket 2 HD` `Jio Cricket 3 HD` `Highbrow` `Nick HD ` `God TV` `Eurosport` `HHDL` `Top Tutor` `RT TV` 
		
		###Nepali
		>`Nepal one` 
		
		###French
		>`France 24` 
		
		###Urdu
		>`Mercy TV` `DD urdu` `DD Kashir` `Zee Salaam` `Channel Win` `Gulistan News` `News18 Urdu` `4 TV` `Tehzeeb TV`"
    
    """
    url = f"https://indian-tv-schedule.p.rapidapi.com/TodaySchedule"
    querystring = {'channel': channel, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-tv-schedule.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettodaysmovies(lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## List of Movies of all channels in a Langauge
		#### Press left three dots for more details
		> **Languages supported**:- (*Keyword for lang*)
		`Hindi` `Tamil` `Telugu` `Marathi` `Kannada` `Malayalam`  `Bengali` `Punjabi` `Urdu` `English`  `Gujarati` `Odia`  `Bhojpuri` `Assamese` `Nepali` `French`"
    
    """
    url = f"https://indian-tv-schedule.p.rapidapi.com/GetTodaysMovies"
    querystring = {'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-tv-schedule.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

