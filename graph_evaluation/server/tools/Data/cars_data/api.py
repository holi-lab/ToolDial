import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_cars_data(min_front_track_in: int=None, max_front_track_in: int=None, min_height_mm: int=None, min_wheelbase_in: int=None, max_max_weight_kg: int=None, min_length_mm: int=None, max_max_load_kg: int=None, min_length_in: int=None, min_width_mm: int=None, min_max_load_lbs: int=None, min_width_in: int=None, min_fuel_tank_capacity_l: int=None, min_height_in: int=None, max_height_mm: int=None, max_fuel_tank_capacity_l: int=None, max_cylinder_bore_mm: int=None, min_max_weight_kg: int=None, min_max_weight_lbs: int=None, min_piston_stroke_in: int=None, max_number_of_valves_per_cylinder: int=None, max_piston_stroke_mm: int=None, max_kerb_weight_lbs: int=None, max_kerb_weight_kg: int=None, max_width_in: int=None, min_kerb_weight_lbs: int=None, max_cylinder_bore_in: int=None, min_number_of_valves_per_cylinder: int=None, min_compression_ratio: int=None, max_compression_ratio: int=None, min_kerb_weight_kg: int=None, max_piston_stroke_in: int=None, min_maximum_speed_mph: int=None, min_horsepower: int=None, max_maximum_speed_km_per_hour: int=None, min_acceleration_0_100_kmh_sec: int=None, min_cylinder_bore_in: int=None, max_start_of_production_year: int=None, max_number_of_cylinders: int=None, max_maximum_speed_mph: int=None, max_acceleration_0_100_kmh_sec: int=None, min_piston_stroke_mm: int=None, min_number_of_cylinders: int=None, max_horsepower: int=None, max_seats: int=None, drive_wheel: str=None, power_steering: str=None, min_end_of_production_year: int=None, min_start_of_production_year: int=None, max_wheelbase_in: int=None, max_wheelbase_mm: int=None, max_rear_back_track_in: int=None, max_length_mm: int=None, max_width_mm: int=None, max_height_in: int=None, min_wheelbase_mm: int=None, max_length_in: int=None, min_front_track_mm: int=None, max_rear_back_track_mm: int=None, min_rear_back_track_mm: int=None, min_rear_back_track_in: int=None, max_front_track_mm: int=None, max_max_load_lbs: int=None, min_max_load_kg: int=None, max_max_weight_lbs: int=None, min_doors: int=None, min_cylinder_bore_mm: int=None, min_maximum_speed_km_per_hour: int=None, max_doors: int=None, min_seats: int=None, max_end_of_production_year: int=None, assisting_systems: str=None, steering_type: str=None, fuel_type: str=None, model: str=None, limit: int=10, skip: int=0, brand: str=None, title: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of over 40k car models through a simple but powerful api. The results will contain plenty of information that will cover your business needs"
    max_max_weight_kg: Maximum value for maximum weight in kilograms
        max_max_load_kg: Maximum value for maximum load in kilograms
        min_max_load_lbs: Minimum value for maximum load in lbs
        min_max_weight_kg: Minimum value for maximum weight in kilograms
        min_max_weight_lbs: Minimum value for maximum weight in lbs
        min_maximum_speed_mph: Minimum value for maximum speed in miles per hour
        max_maximum_speed_km_per_hour: Maximum value for maximum speed in km per hour
        max_start_of_production_year: Maximum value for start_of_production_year
        max_maximum_speed_mph: Maximum value for maximum speed in miles per hour
        drive_wheel: Filter by drive_wheel.

**Possible values are:**
All wheel drive (4x4),Front wheel drive,Rear wheel drive
        power_steering: Filter by power_steering

**Possible values are:**
Electric Steering,Hydraulic Steering

        min_end_of_production_year: Minimum value for end_of_production_year
        min_start_of_production_year: Minimum value for start_of_production_year
        max_max_load_lbs: Maximum value for maximum load in lbs
        min_max_load_kg: Minimum value for maximum load in kilograms
        max_max_weight_lbs: Maximum value for maximum weight in lbs
        min_maximum_speed_km_per_hour: Minimum value for maximum speed in km per hour
        max_end_of_production_year: Maximum value for end_of_production_year
        assisting_systems: Filter by assisting_systems

**Possible values are:**
4-wheel steering (4WS, active rear steering),ABS (Anti-lock braking system),ABS (Anti-lock braking system)4-wheel steering (4WS, active rear steering)
        steering_type: Filter by steering_type

**Possible values are:**
Cone worm with recirculation balls,Steering rack and pinion,Worm-reduction unit
        fuel_type: Filter by the fuel type. 

**Possible values are:**
Diesel,Electricity,Ethanol - E85,Hydrogen,Hydrogen / electricity,LPG,Mixture of two stroke engine,Petrol (Gasoline),Petrol / CNG,Petrol / Ethanol - E85,Petrol / LPG,diesel / electricity,petrol / Ethanol - E85 / electricity,petrol / electricity
        model: The model of the car.

**Possible values are:**
/8,#1,001,003,009,02,05,1,1 Series,1.0,1.1,10,100,100 NX,1000,1007,101,104,105,120,106,107,108,11,110,1102,1103,1105,1111,1111 Oka,1125,11301,114,1140,117,118,12,121,124,124 Spider,125p,126,126p,127,127p,128,128 Skala,13,130,1300,1307-1510,131,1310,132,1325,132p,133,14,140,1410,145,146,147,15,1500,155,156,159,16,160,1600,164,166,17,1750-2000,18,180,180 SX,1800,181,19,190,2,2 CV,2 Series,2-Eleven,2.5 PI MK,2.8 - 5.3,20,200,200 SX,2000 MkII,2000-3500 Hatchback,2008,204,205,206,207,208,208/308,21,210,2101,2102,2103,2104,2105,2106,2107,2108,2109,2110,2111,2112,2113,2114,2115,2120 Nadezhda,2123,2125,2126,21261,2129,2131,2136 Kombi,2137 Kombi,2138,2140,2141,21C,22,2200-3500,228,2328,2329,238,24,240,240SX,25,250 GT,250 GTO,2500,2500 GT,2500/3500,260,275,280,280 Zx,zxt,296,3,3 Series,3 Wheeler,3-Eleven,3.0,3.0 CSL,30,300,300 SLR,300 ZX,3000,3000 GT,3008,300M,301,304,305,306,307,308,309,31,311,3151,31512,31514,31519,3153,3159,3160,3162,3200 GT,323,328,33,330,340-360,348,350,350 GT,3500 GT,350Z,353,356,360,365,370Z,375,390,4,4 Series,4.0,4/4,400,400 GT,4007,4008,401,402,403,404,405,406,407,408,408 (crossover),4104,412,420,420/430,423 Kombi,427,434,440 K,45,450,456,458,46,460 L,469,480 E,488,4C,4runner,5,5 Series,5.0,50,500,5000,5000 GT,5008,504,505,508,5300 GT Strada,540C,550,57,57 S,570S,575M Maranello,580,595,599,6,6 Series,6.0,600,6000,600LT,604,605,607,612 Scaglietti,616,62,62 S,620R,625C,626,6500 (Land King),650S,66,660 LMS,675LT,69,695,7,7 Series,7.0,700,718,720S,730,740,75,750 Monza,750S,760,765LT,780 Bertone,8 Series,80,800,806,807,812,818 Kombi,850,86,8C Competizione,9,9-2X,9-3,9-4X,9-5,9-7X,90,900,900   T/E,9000,911,912,914,917,918,924,928,929,940,944,95,950,959,96,960,965,966,968,969,99,9x8,A 112,A-class,A1,A110,A2,A25,A3,A310,A4,A5,A6,A6 e-tron,A610,A7,A8,AD,AIV Roadster,AM4,AMG GT,AMG GT 4-Door Coupe,AMG ONE,AMI,AMV,AMV8,ASX,ATS,AX,Abruzzi,Acadia,Acadiane,Accent,Acclaim,Accord,Achieva,Across,Actyon,Adam,Admiral,Adventurer,Aerio,Aero 8,Aerostar,Agera,Agila,Agile,Air,Airtrek,Airwave,Alaskan,Albaycin,Albea,Alcazar,Alero,Alfasud,Alfetta,Alhambra,Allante,Allegro,Allex,Allion,Almera,Alpha-S,Alpha-T,Alphard,Altea,Altezza,Altima,Altis,Alto,Altroz,Alturas G4,Alza,Amarok,Amaze,Ambassador,Ambulance,Ameo,Ampera,Amulet,Antara,Antelope,Apollo,Applause,Aqua,Aquila,Arcadia,Arena,Argenta,Argo,Aria,Aristo,Ariya,Arkana,Armada,Arna,Arnage,Arona,Arosa,Arrizo 5,Arrizo GX,Arteon,Artura,Aruz,Ascender,Ascent,Ascona,Aska,Aspen,Aspire,Assol,Asterion,Astor,Astra,Astro,Ateca,Atenza,Ativa,Atlas,Atlas Pro,Atom,Atos,Atrai/extol,Atto 3,Attrage,Audi 100,Aura,Aurion,Auris,Aurora,Austral,Avalanche,Avalon,Avancier,Avante,Avanti,Avantime,Avanza,Avella,Avenger,Avenir,Avensis,Avensis Verso,Aventador,Aveo,Aviator,Axel,Axela,Axia,Axiom,Aygo,Aygo X,Ayla,Az-1,Az-offroad,Az-wagon,Aztec,Azure,B-MAX,B-class,B-series,B1,B10,B11,B12,B3,B4,B5,B6,B7,B8,B9,BD,BJ 2020,BLS,BR-V,BRZ,BS4,BS6,BT-50,BT62,BX,BX5,BX6,BX7,BXi7,Baja,Baleno,Barchetta,Barchetta Stradale,Barina,Bassara,Bayon,Be-go,Beauty Leopard,Beetle,Bel Air,Belta,BenBen,Bentayga,Beretta,Berlinetta Boxer,Berlinette,Berlingo,Besturn B30,Besturn B50,Besturn B70,Besturn B90,Besturn X40,Besturn X80,Beta,Bezza,Biante,Bighorn,Bigua,Binyue,Bipper,Bistro,Biturbo,Black Cuillin,Blade,Blazer,Blenheim,Blizzard,Bluebird,Bo Rui GE,Bolero,Bolide,Boliger,Bolt,Bolt EUV,Bolt EV,Bongo,Bonneville,Boon,Bora,Born,Borrego,Boxster,Brasilia,Brava,Bravada,Bravo,Breeze,Brera,Brevis,Brio,Bronco,Bronco Sport,Brooklands,Brougham,Budii,Bulldog,Bullet,C+pod,C-10,C-Crosser,C-Elysee,C-HR,C-MAX,C-X16,C-Zero,C-class,C1,C12,C2,C3,C3-XR,C30,C32,C4,C4 X,C40,C5,C5 Aircross,C5 X,C51,C52,C6,C61,C70,C8,C81,C9,CC,CC850,CJ,CJ 3,CJ 3 Wagon,CL,CLA,CLC,CLE,CLK,CLK GTR,CLS,CR-V,CR-Z,CRX,CS15,CS35,CS55,CS85,CS95,CSR,CT,CT4,CT5,CT6,CTR,CTS,CTwo,CX,CX-3,CX-30,CX-4,CX-5,CX-50,CX-60,CX-7,CX-8,CX-9,CX-90,CX70,Cabriolet,Caddy,Cadenza,Calais,Caldina,Caliber,Calibra,California,Calya,Camaro,Cami,Campo,Camry,Canyon,Capa,Capella,Capital,Cappuccino,Capri,Caprice,Captiva,Captur,Caravan,Carens,Carib,Carina,Carisma,Carlton Mk,Carmen,Carnival,Carol,Carrera GT,Cascada,Cat,Catera,Cavalier,Cayenne,Cayman,Cedric,Cee'd,Cefiro,Celebrity,Celerio,Celesta,Celeste,Celica,Celsior,Celta,Centenario,Centennial,Centodieci,Century,Cerato,Cerbera,Ceria,Chairman,Challenger,Chance,Chaos,Charade,Charger,Chariot,Charmant,Chaser,Cherokee,Cherry,Chevette,Chimaera,Chiron,Chubasco,Ciaz,Cima,Cinquecento,Cirrus,Citan,Citigo,City,City EV,City Golf,City K-ZE,City Leading,CityCross,Civic,Civic Type R,Clarity,Clarus,Classic,Clef,Clio,Club,Clubman,Clubsport,Cobalt,Coffee,Colorado,Colt,Colt Lancer,Combo,Commander,Commendatore,Commodore,Compass,Concerto,Concord,Concorde,Conquest,Consul,Contessa,Continental,Continental Mark,Contour,Convertible,Cool Bear,Coolray,Copen,Corbellati,Cordia,Cordoba,Corniche,Corolla,Corolla Cross,Corolla Rumion,Corolla Spacio,Corolla Verso,Corona,Corrado,Corsa,Corsair,Corsica,Corvette,Cosmo,Cougar,Countach,Countryman,Coupe,Cowboy,Cowry,Cressida,Cresta,Crew,Croma,Cronos,Cross4,Crossblade,Crossfire,Crossland,Crossroad,Crosstour,Crosstrek,Crown,Crown Majesta,Crown Victoria,Cruze,Cube,Cullinan,Cultus,Cuore,Curren,Custom,Cutlass,Cygnet,Cynos,D-Max,D1,D10,D2S,D3,D4,D5,D60,D8,D90,DB11,DB12,DB4,DB5,DB6,DB7,DB9,DBS,DBX,DS,DS3,DS4,DS5,DTS,Daimler,Dakota,Damas,Dargo,Dart,Datsun,Dauphine,Dawn,Daytona,Daytona SP3,Daytona Shelby,DeVille,Debonair,Dedra,Deer,Defender,Delica,Delta,Deluxe,Demio,Derby,Diablo,Diamante,Dingo,Dino,Dion,Diplomat,Discovery,Discovery Sport,Divo,Dmc-12,Doblo,Dogan,Dokker,Dolomite,Dolphin,Domani,Domingo,Duet,Duna,Durango,Duster,Dyane,Dynasty,Dzire,E 1600,E Verito,E-HS9,E-Pace,E-Racer,E-class,E-series,E-type,E100,E200,E70,E80,E9,EB 110,EB 112,EC6,EC7,EF7,EL7,ELR,EMOTION,EMotion,EON,EP9,EQ fortwo,EQA,EQB,EQC,EQE,EQE SUV,EQS,EQS SUV,EQV,ES,ES6,ES7,ES8,ET5,ET7,EV K12,EV K17,EV6,EVO37,EVO5,EX,EX30,EX90,Eado,Echo,Eclat,Eclipse,Eclipse Cross,EcoSport,Econoline,Edge,Edix,Edonis,Egea,Eighty-Eight,Elan,Elan Sport,Elantra,Elba,Eldorado,Electra,Electra E4,Electra E5,Electric,Element,Eletre,Elevate,Elgrand,Elise,Elite,Elva,Elysion,Emeraude,Emgrand EV,Emgrand GL,Emgrand GS,Emgrand GT,Emgrand X7,Emira,Enclave,Encore,Encore GX,Endeavor,Endurance,Enterprise,Envision,Envista,Envix,Envoy,Enyaq iV,Enzo,Eos,Epica,Equator,Equinox,Equus,Ertiga,Escalade,Escape,Escort,Escudo,Espace,Espada,Esperante,Espero,Esprit,Esse,Essenza SCV12,Estafette,Estate,Esteem,Estima,Etios,Etos,Eunos 500,Eunos 800,Eunos Cosmo,Europa,Evanda,Evantra,Evasion,Everest,Every,Every Landy,Evija,Evora,Excel,Excelle,Excelle GT,Excursion,Exeo,Exige,Exora,Expedition,Expert,Explorer,Express,F-250 Super Duty,F-350 Super Duty,F-450 Super Duty,F-Pace,F-Series F-100/F-150,F-Series F-250,F-type,F.F SUPERQUAD,F0,F1,F12,F3,F35,F355,F40,F430,F5,F50,F6,F7,F8,FC,FF,FIT,FJ Cruiser,FLYER II,FR-S,FR-V,FTO,FX,FXE,FXX,Fabia,Fadil,Fairlady,Falcon,Familia,Family,Favorit,Felicia,Fengon E3,Fengshen A9,Fenyr,Feroza,Festiva,Fiero,Fiesta,Fifth Avenue,Figaro,Fighter,Figo,Fintail,Fiorino,Firebird,Firedome,Fireflite,Firenza Coupe,Firesweep,Fit Aria,Five Hundred,Flavia,Fleetwood,Flex,Florid,Fluence,Flying,Flying Spur,Focus,Fora,Forenza,Forester,Forfour,Formentor,Forte,Fortuner,Fortwo,Fox,Freeclimber,Freed,Freelander,Freemont,Freestar,Freestyle,Frontera,Frontier,Frontlander,Fuego,Fuga,Fullback,Fulvia,Fun,Funcargo,Fura,Fusion,G,G-class,G-modell,G10,G20,G2X,G3,G40,G50,G55,G6,G70,G8,G80,G9,G90/EQ900,GA4,GC6,GL,GL6,GL8,GLA,GLB,GLC,GLE,GLK,GLS,GM8,GO,GS,GS3,GS4,GS5,GS8,GT,GT-R,GTA,GTA Coupe,GTC4Lusso,GTE,GTO,GTS,GTSR,GTV,GV60,GV70,GV80,GX,Gaia,Galant,Galaxy,Gallardo,Galloper,Galue,Gamma,Gemera,Gemini,Genesis,Geneva,Gentra,Geo Storm,Getz,Ghibli,Ghost,Giulia,Giulietta,Gladiator,Glanza,Gloria,Gloster,Goa,Gol,Golf,GranCabrio,GranSport,GranTurismo,Granada,Grand AM,Grand Albaycin,Grand Cherokee,Grand Commander,Grand Escudo,Grand Hiace,Grand Marquis,Grand Prix,Grand Vitara,Grand Voyager,GrandTiger,Grandeur/Azera,Grandis,Grandland,Granta,Granvia,Grecale,Grenadier,Griffith,Groove,Guara,Gurkha,Gypsy,H-1,H1,H2,H3,H4,H5,H6,H7,H9,HHR,HK GT,HR-V,HRV Excelle,HS,HS5,HS7,Han,Haoyue,Harrier,Hatch,Hexa,Hi-topic,Hiace,Highlander,Hilux,Himiko,Hoggar,Horizon,Hornet,Hover CUV,Hover H5,Hover H6,Huayra,Hunter,Huracan,Hustler,Hyena,I-Pace,I30,I35,ID,ID. 2all,ID. Buzz,ID. CROZZ,ID. VIZZION,ID.3,ID.4,ID.5,ID.6,IE,ION,IONIQ,IONIQ 5,IONIQ 6,IS,ISis,Ibiza,Icon,Idea,Ignis,Iltis,Impala,Imperator,Imperio,Impreza,Impulse,Inca,Indica,Indigo,Indy,Innova,Insight,Insignia,Inspire,Integra,Intrepid,Intrigue,Ipsum,Iriz,Isetta,Islero,Ist,Istana,Ixion,Izoa,J30,Jackaroo,Jade,Jalpa,Jarama,Jazz,Jesko,Jetta,Jia Ji,Jimmy,Jimny,Jinn,Jogger,Joice,Jolion,Journey,Juara,Juke,Jumpy,Justy,K2,K3,K4,K5,K50,K7,K8,K9,KA,KUV100,KWID,KX3,Kadett,Kadjar,Kaefer,Kalina,Kallista,Kalos,Kamiq,Kangaroo,Kangoo,Kappa,Kaptur,Karif,Karl,Karma,Karmann Ghia,Karoq,Kartal,Kei,Kelisa,Kenari,Khamsin,Kicks,Kiger,Kimo (A1),King Cab,Kizashi,Kluger,Knyaz Vladimir,Kodiaq,Koleos,Komendant,Kona,Kondor,Korando,Korando Sports,Kuga,Kushaq,Kyalami,Kyron,L200,L5,LC,LE Baron,LE Mans,LE Sabre,LF-Z,LFA,LFT-666,LHS,LM,LM 400,LM 500,LM001,LM002,LMA002,LN,LS,LSE,LUV D-MAX,LUX A,LUX SA,LW,LX,LYRIQ,La Joya,LaCrosse,LaFerrari,Lacetti,Lafesta,Lagonda,Lagreat,Laguna,Lamando,Lancer,Lancer Evolution,Land Crown,Land Cruiser,Land Cruiser Prado,Landaulet,Landaulette,Landmark,Landscape,Landtrek,Langley,Lanos,Lanos (Sens),Lantis,Lantra,Laputa,Largo,Largus,Laser,Latitude,Laurel,Lavida,Lavita,Le-Seyde,LeMans,Leaf,Leeza,Legacy,Leganza,Legend,Legnum,Leon,Leone,Leopard,Levante,Levin,Levorg,Liana,Libero,Liberty,Life,Lightning,Linea,Linmax,Lioncel,Lite Ace,Livina,Lodgy,Logan,Logistar 100,Logo,Lova,Lucerne,Lucino,Luka EV,Lumin,Lumina,Lupo,Lybra,Lykan,M,M-20,M-class,M1,M12,M12 GTO,M2,M3,M3e,M4,M5,M6,M600,M7,M8,MAX,MC12,MC20,MDX,MG3,MG4,MG5,MG6,MGB,MGF,MGR,MINAUTO,MK,MKC,MKS,MKT,MKX,MKZ,MK_1,MM 540/550,MM 775,MM Double Cab,MP4-12C,MPV,MPV EV,MR 2,MR Wagon,MR-S,MS-8,MU-X,MX-3,MX-30,MX-5,MX-6,MX3,Macan,Maestro,Magentis,Magnite,Magnum,Magnus,Maguari HS1,Malaga,Malibu,Mangusta,Manta,Mantis,Marauder,Marazzo,Marbella,March,Marea,Marina,Mariner,Mark,Mark II,Mark LT,Mark X,Marshal,Marvel R,Marvel X,Marzal,Massif 4x4,Master,MasterAce,Materia,Matiz,Matra Bagheera,Matrix,Maverick,Maxi,Maxima,Mega Cruiser,Megane,Megane E-Tech Electric,Menlo,Merak,Meriva,Metro,Mexico,MiTo,Micra,Microlino,Midget,Mifa 9,Mii,Milan,Mille,Millenia,Mini,Mini MK,Mini Remastered,Minica,Mint,Mira,Mira Gino,Mirage,Mirai,Mission E,Mistral,Miura,Mm550 DP,Mobilio,Moco,Model 3,Model S,Model X,Model Y,Modus,Mohave,Mokka,Monaco,Monaro,Mondeo,Mondial,Monjaro,Mono,Montana,Monte Carlo,Montecarlo,Montego,Monterey,Montero,Montero Sport,Montreal,Monza,Mountaineer,Movano,Move,Mu,Mulan,Mulsanne,Multipla,Multivan,Murano,Murcielago,Murena,Musa,Musso,Mustang,Mustang Mach-E,Mystique,Myvi,N-Box,N-One,N-WGN,NC 640 DP,NF,NP 300 Pick up,NSU RO 80,NSX,NV200,NX,Nadia,Naked,Nano,Nassau,Nautilus,Navajo,Navara,Navigator,Neon,Nevera,New Class,New Yorker,Nexa,Nexia,Nexo,Nexon,Nippa,Niro,Nitro,Niva,Nivus,Noah,Nomad,Note,Nova,Nubira,NuvoSport,OWL,Oasis,Ocean,Octavia,Odyssey,Okavango,Omega,Omega Caravan,Omni,Omoda 5,One,One-77,One:1,Onix,Opa,Opirus,Opti,Optima,Origin,Orion,Orlando,Orthia,Otaka,Oting,Ottimo,Ousado,Outback,Outlander,Outlook,P 601,P1,P5,P7,P8,PB18,PICKUP X3,POER,PT Cruiser,Paceman,Pacifica (crossover),Pacifica (minivan),Pagoda,Pajero,Pajero Sport,Palio,Palisade,Panamera,Panda,Park Avenue,Park Ward,Partner,Paseo,Passat,Passat CC,Passo,Passport,Pathfinder,Patriot,Patrol,Pegasus,Perdana,Peri,Persona,Persona 300 Compact,Phaeton,Phantom,Phedra,Phideon,Phoenix,Piazza,Picanto,Pick UP,Pick Up,Pickup,Picnic,Pilot,Pinzgauer,Pistachio,Pixo,Platz,Pleo,Plus 4,Plus 8,Plus Four,Plus Six,Plutus,Pointer,Polo,Polo Vivo,Polonez,Ponton,Pony,Porte,Porter,Portofino,Potentia,Powermaster Six,Prairie,Praktik,Prelude,Premacy,Premier,Premio,Presage,Presea,President,Preve,Previa,Pride,Primera,Prince,Princess,Princip,Priora,Prisma,Prius,Prizm,Pro Cee'd,Proace,Proace City,Probe,Probox,Proceed,Progres,Project Black S,Pronard,Protege,Proudia/dignity,Prowler,Pulsar,Pulse,Puma,Punch,Punto,Purosangue,Pyzar,Q2,Q3,Q30,Q4 e-tron,Q45,Q5,Q50,Q60,Q7,Q70,Q8,Q8 e-tron,QM3,QM6,QQ6 (S21),QX30,QX4,QX50,QX55,QX56,QX60,QX70,QX80,Qashqai,Quanto,Quattro,Quattroporte,Qubo,Quest,Quintet,Quoris,R Nessa,R-class,R1,R1T,R2,R4,R42,R8,RAM,RAV4,RC,RC-5,RCZ,RLE Roadster,RM-5,RS 2,RS 3,RS 4,RS 5,RS 6,RS 7,RS Q3,RS Q8,RS e-tron GT,RS-3,RS-5,RS2000,RUV,RVR,RX,RX-3,RX-7,RX-8,RX3,RX5,RX8,RXR One,RZ,Racer,Ractis,Raeton,Rafaga,Rafale,Raider,Rainier,Raize,Rally 037,Ramcharger,Rancho,Range Rover,Range Rover Evoque,Range Rover Sport,Range Rover Velar,Ranger,Rapid,Rapide,Rasheen,Raum,Reatta,Red,Refine,Regal,Regata,Regera,Regius,Rein,Reina,Rekord,Relay,Ren,Rendezvous,Renegade,Reno,Retona,Reventon,Revero,Revolution,Revue,Revuelto,Rexton,Rezzo,Rich,Ridgeline,Rifter,Rio,Ritmo,Riviera,Road Partner,Roadmaster,Roadster,Roadster S,Roadster V8,Rock Star,Rocky,Rocsta,Rodeo,Rodius,Rogue,Rogue Sport,Roma,Ronda,Roomster,Routan,Roxor,Royale,Rp1,RtR,Rush,Ryoga,S,S-10 Pickup,S-Coupe,S-MAX,S-MX,S-Presso,S-class,S-type,S1,S2,S2000,S3,S4,S40,S5,S5 Young,S6,S60,S600,S660,S7,S70,S8,S80,S800,S90,SA01,SC,SCR,SENSATION,SF90,SL,SLC,SLK,SLR McLaren,SLS AMG,SM,SM3,SM5,SM6,SM7,SQ2,SQ5,SQ7,SQ8,SQ8 e-tron,SR-V X3,SRX,SS,SSK,SSR,ST1,STS,SUV,SUV X3,SVX,SW,SX4 S-Cross,SZ,Saber,Sable,Safari,Safe,Safrane,Saga,Saga Iswara,Sagaris,Sahin,Saibao,Sail/S-RV,Sailor,Sakura,Saladin,Saloon,Samand,Samba,Samurai,Sandero,Santa Cruz,Santa Fe,Santa Fe Classic,Santamo,Santana,Santro,Sapporo,Saratoga,Sarthe,Savana,Saveiro,Saxo,Scala,Scalo,Scenic,Sceo,Scepter,Scimitar Sabre,Scirocco,Scorpio,Scrum,Seagull,Seal,Sebring,Sedici,Sedona,Seicento,Seltos,Senat,Senator,Senia R7,Senna,Senova D50,Senova X55,Sens,Sentia,Sentra,Sephia,Sequoia,Sera,Serce,Serena,Seres 3,Series I,Series II,Seven,Seville,Shadow,Shamal,Sharan,Shelby,Shogun,Shuma,Shuttle,Sian FKP 37,Siber,Sibylla,Sidekick,Siena,Sienna,Sienta,Sierra,Sierra 1500,Sierra 2500HD,Sierra 3500HD,Sigma,Signum,Sigra,Silhouette,Silver Dawn,Silver Seraph,Silver Spur,Silverado 1500,Silverado 2500 HD,Silverado 3500 HD,Silvia,Simbo,Simca,Sintra,Sion,Sirion,Sixteen,Sky,Skylark,Skyline,Slavia,Slingshot,Small,Smart,Smoothing,SoCool,Soarer,Solara,Solaris,Solenza,Solstice,Solterra,Soluto,Sonata,Sonet,Song,Song Max,Sonic,Sonica,Sonoma,Soren,Sorento,Soul,Space Gear,Space Runner,Space Star,Space Wagon,SpaceTourer,Spark,Sparky,Spartana,Spectra,Spectrum,Speedback GT,Speedback Silverstone Edition,Speedster,Speedtail,Spiano,Spider,Spin,Spirit,Spitfire,Splash,Sport EV,Sport Spider,Sport Trac,Sportage,Spring,Sprinter,Spyder,Stag,Stagea,Stanza,Starcraft,Staria,Starion,Starlet,Statesman,Stealth,Steed,Stella,Stellar,Stelvio,Stepwgn,Stilo,Stinger,Stonic,Storia,Storm,Strada,Stradale,Stratus,Stream,Streetka,Stylus,Suburban,Succeed,Summit,Sumo,Sunbird,Sundance,Sunfire,Sunny,Super,Super 3,Super 5,Super Eight,Superb,Supra,Suprima S,Supro,Swace,Sweet (QQ),Swift,Syclone,Sylphy,Symbol,T-Cross,T-REX,T-Roc,T-class,T.33,T.50,T.50s,T3,T300,T60,T600,T613,T700,T77,T90,TCR,TF,TF 1800,TF 2000 MK1,TG,TR 6,TR 7,TR 8,TS,TS1,TSR,TSR-S,TT,TUV300,TX,Tacoma,Tacqua,Tacuma,Tager,Tagora,Tahoe,Taigo,Taigun,Taimar,Talagon,Talento,Taliant,Talisman,Talon,Tamora,Tang,Tank 300,Tanto,Taos,Taro,Tarraco,Tasmin,Taunus,Taurus,Taurus X,Tavascan,Tavera,Taxi,Taycan,Tayron,Teana,Telluride,Tempo,Tempra,Teramont,Tercel,Terios,Terra,Terracan,Terrain,Terrano,Terraza,Territory,Testarossa,Thar,Tharu,That S,Thema,Thesis,Thunderbird,Tiago,Tiburon,Tickford Capri,Tico,Tiggo,Tiggo 3,Tiggo 5,Tiggo 5x,Tiggo 7,Tiggo 8,Tiggo 9,Tigor,Tigra,Tiguan,Tiida,Tino,Tipo,Titan,Tivoli,Today,Toledo,Tonale,Tonic,Topaz,Toppo,Torneo,Toro,Torrent,Torres,Tosca,Touareg,Touran,Tourneo Connect,Tourneo Courier,Tourneo Custom,Town & Country,Town Ace,Town BOX,Town Car,Townstar,Tracer,Track,Tracker,Trafic,Trailblazer,Trajet,Trans Sport,Traveller,Traverse,Traviq,Trax,Tredia,Trevi,Trevis,Trezia,Tribeca,Triber,Tribute,Triton,Trooper,Tuatara,Tucson,Tugella,Tundra,Turbo R,Tuscan,Tuscan Challenge,Tuscani,Twingo,Twizy,Type 3,Type 57,Typhoon,U5,U6,U7,UNI-K,UNI-T,UNI-V,UNO,UR-V,UX,Ultimate Aero,Ulysse,Up!,Uplander,Urban Cruiser,Urban Cruiser Hyryder,UrbanRebel,Urraco,Urus,Ute,Utopia,V-class,V12 Vantage,V13R,V16t,V3,V40,V50,V60,V7,V70,V8,V8 Vantage,V90,VE-1,VF8,VF9,VISION EQXX,VUE,VV5,VV7,VX,VX220,VXR8,Vallelunga,Vamos,Van,Vaneo,Vanette,Vanguard,Vanquish,Vantrend,Vectra,Vega,VehiCross,Vel Satis,Velite 6,Velite 7,Veloster,Veneno,Venere,Venga,Venice,Venom F5,Venom GT,Vento,Ventora,Venture,Venue,Venza,Veracruz,Verisa,Verito,Verito Vibe,Verna,Vero,Verona,Verossa,Versa,Verso,Verso-S,Vespacar,Vesta,Veyron,Vezel,Viaggio,Viano,Vibe,Viceroy,Victor,Vigor,Villager,Viloran,Vios,Viper,Virage,Virtus,Visa,Vision,Vision-S,Vista,Visto,Vita,Vitara,Vitara Brezza,Vito,Vitz,Viva,Vivaro,Vivio,Voleex,Volt,Voltz,Vortex Estina,Voxy,Voyager,W100,W108,W109,W111,W112,W12,W123,W124,W136,W150,W187,W188,W29,W8 Twin Turbo,WR-V,WRX,Wagon R,Wagon R+,Wagoneer,Waja,Warszawa,Westfield,Wigo,Wildlander,Will,Wind,Windom,Windstar,Wingle,Wingroad,Winstorm,Wish,Wizard,Wraith,Wrangler,X,X 1/9,X-90,X-Bow,X-Trail,X-class,X-type,X1,X2,X3,X3 M,X4,X4 M,X5,X5 M,X50,X6,X6 M,X7,X7 Sport,X70,XB7,XC40,XC60,XC70,XC90,XCeed,XD3,XD4,XE,XF,XG,XJ,XJ 40, 81,XJ220,XJS,XK,XL1,XL7,XLR,XLV,XM,XRAY,XS,XT,XT4,XT5,XT6,XTR,XTS,XUV500,XUV700,XV,Xantia,Xbee,Xcent,Xedos 6,Xedos 9,Xenia,Xenon,Xiaoyao,Xingrui,Xingyue,Xingyue L,Xpander,Xpower SV,Xsara,Xterra,Xylo,YRV,Yaris,Yaris Cross,Yeti,Ypsilon,Yuan,Yugo,Yuhu,Yukon,Yuriy Dolgorukiy,Yusheng,Z,Z-Chine,Z1,Z100,Z3,Z300/Z360,Z4,Z560,Z8,ZND,ZR,ZR-V,ZS,ZT,ZX,Zafira,Zafira Life,Zagato,Zanturi,Zeclat,Zen,Zephyr,Zero,Zerouno,Zest,Zeta,Zoe,Zonda,bB,bZ4X,cB7,e,e-LEGEND,e-Mehari,e-tron,e-tron GT,e2,e3,e6,e:Ny1,eAIXAM,eK,eK X,i,i-MiEV,i10,i20,i3,i30,i4,i40,i5,i6,i7,i8,iA,iM,iMAX8,iQ,iV6,iX,iX1,iX3,ix20,ix25/Creta,ix35,ix55,mi-DO,nanuk quattro concept,on-DO,tC,xA,xB,xD
        limit: Limit number of results (between 1 and 100). Default set to 10
        skip: Skip first N results where N is a positive value.
        brand: The brand of the car

**Possible values are:**
AITO,Abarth,Aiways,Aixam,Alfa Romeo,Alpina,Alpine,Anfini,Apollo,Arcfox,Aria,Ariel,Aro,Artega,Asia,Aspark,Aston Martin,Astro,Audi,Aurus,Austin,Austin-Healey,Autobianchi,Avatr,B.Engineering,BAC,BAIC Motor,BMW,BYD,Baltijas Dzips,Baojun,Bee Bee,Bentley,Bertone,Bestune,Bisu,Bitter,Bizzarrini,Blonell,Bollinger,Bordrin,Borgward,Brabham,Bremach,Brilliance,Bristol,Bufori,Bugatti,Buick,Cadillac,Callaway,Campagna,Carbodies,Caterham,Cenntro,ChangAn,ChangFeng,Chery,Chevrolet,Chrysler,Citroen,Cizeta,Corbellati,Cupra,Czinger,DAF,DC,DFSK,DR,DS,Dacia,Dadi,Daewoo,Daihatsu,Daimler,Dallara,Dallas,Datsun,David Brown,De Lorean,De Tomaso,DeSoto,Derways,Dodge,DongFeng,Doninvest,Donkervoort,Drako,EVO,Eadon Green,Eagle,Elemental,Engler,FAW,FOMM,FSO,Felino,Ferrari,Fiat,Fisker,Fittipaldi,Force Motors,Ford,Fuqi,GAZ,GFG Style,GMC,Geely,Genesis,Geo,Ginetta,Gleagle,Gordon Murray,Great Wall,HSV,Hafei,Haima,Haval,Hawtai,Hennessey,Hindustan,Hispano Suiza,Holden,Honda,Hongqi,HuangHai,Hummer,Hurtan,Hyundai,ICKX,IMSA,INEOS,Infiniti,Innocenti,Invicta,Invicta Electric,Iran Khodro,Irmscher,Isdera,IsoRivolta,Isuzu,Italdesign,Iveco,Izh,JAC,Jaguar,Jeep,Jiangling,KTM,Karlmann King,Karma,Kia,Kimera,Koenigsegg,LTI,LUAZ,Lada,Lamborghini,Lancia,Land Rover,Landwind,Lexus,Lincoln,Lister,Lordstown,Lotus,Lucid,Luxgen,Lvchi,MCC,MG,MINEmobility,MW Motors,Mahindra,Marcos,Maruti,Maserati,Maxus,Maybach,Mazda,Mazzanti,McLaren,Mega,Melkus,Mercedes-Benz,Mercury,Metrocab,Micro,Milan,Minelli,Mini,Mitsubishi,Mitsuoka,Moke,Monte Carlo,Morgan,Morris,Moskvich,Munro,NIO,Nissan,Noble,O.S.C.A.,ORA,Oldsmobile,Opel,PUCH,Pagani,Panoz,Paykan,Perodua,Peugeot,Picasso,Pininfarina,Plymouth,Polaris,Polestar,Pontiac,Porsche,Praga,Premier,Proton,Puma,Qiantu,Qoros,RAM,RUF,Ravon,Reliant,Renault,Renault Samsung,Rimac,Rinspeed,Rivian,Roewe,Rolls-Royce,Ronart,Rover,SCG,SMA,SSC,Saab,Saleen,Saturn,Scion,SeAZ,Seat,Seres,ShuangHuan,Sin Cars,Skoda,Smart,Sono Motors,Sony,Soueast,Spectre,Sportequipe,Spyker,Spyros Panopoulos,SsangYong,Subaru,Suda,Suzuki,TVR,TagAz,Talbot,Tata,Tatra,Techrules,Tesla,Tianma,Tianye,Tofas,Tonggong,Toyota,Trabant,Tramontana,Triumph,Trumpchi,UAZ,Uniti,VUHL,VW-Porsche,Vanderhall,Vauxhall,Vector,Vencer,Venturi,Vespa,VinFast,Volkswagen,Volvo,W Motors,WEY,Wartburg,Westfield,Wiesmann,XPENG,Xin Kai,ZAZ,ZIL,ZX,Zacua,Zastava,Zeekr,Zenvo,Zhidou,Zotye,e.GO
        title: The title of the car record
        
    """
    url = f"https://cars-data3.p.rapidapi.com/cars-data"
    querystring = {}
    if min_front_track_in:
        querystring['min_front_track_in'] = min_front_track_in
    if max_front_track_in:
        querystring['max_front_track_in'] = max_front_track_in
    if min_height_mm:
        querystring['min_height_mm'] = min_height_mm
    if min_wheelbase_in:
        querystring['min_wheelbase_in'] = min_wheelbase_in
    if max_max_weight_kg:
        querystring['max_max_weight_kg'] = max_max_weight_kg
    if min_length_mm:
        querystring['min_length_mm'] = min_length_mm
    if max_max_load_kg:
        querystring['max_max_load_kg'] = max_max_load_kg
    if min_length_in:
        querystring['min_length_in'] = min_length_in
    if min_width_mm:
        querystring['min_width_mm'] = min_width_mm
    if min_max_load_lbs:
        querystring['min_max_load_lbs'] = min_max_load_lbs
    if min_width_in:
        querystring['min_width_in'] = min_width_in
    if min_fuel_tank_capacity_l:
        querystring['min_fuel_tank_capacity_l'] = min_fuel_tank_capacity_l
    if min_height_in:
        querystring['min_height_in'] = min_height_in
    if max_height_mm:
        querystring['max_height_mm'] = max_height_mm
    if max_fuel_tank_capacity_l:
        querystring['max_fuel_tank_capacity_l'] = max_fuel_tank_capacity_l
    if max_cylinder_bore_mm:
        querystring['max_cylinder_bore_mm'] = max_cylinder_bore_mm
    if min_max_weight_kg:
        querystring['min_max_weight_kg'] = min_max_weight_kg
    if min_max_weight_lbs:
        querystring['min_max_weight_lbs'] = min_max_weight_lbs
    if min_piston_stroke_in:
        querystring['min_piston_stroke_in'] = min_piston_stroke_in
    if max_number_of_valves_per_cylinder:
        querystring['max_number_of_valves_per_cylinder'] = max_number_of_valves_per_cylinder
    if max_piston_stroke_mm:
        querystring['max_piston_stroke_mm'] = max_piston_stroke_mm
    if max_kerb_weight_lbs:
        querystring['max_kerb_weight_lbs'] = max_kerb_weight_lbs
    if max_kerb_weight_kg:
        querystring['max_kerb_weight_kg'] = max_kerb_weight_kg
    if max_width_in:
        querystring['max_width_in'] = max_width_in
    if min_kerb_weight_lbs:
        querystring['min_kerb_weight_lbs'] = min_kerb_weight_lbs
    if max_cylinder_bore_in:
        querystring['max_cylinder_bore_in'] = max_cylinder_bore_in
    if min_number_of_valves_per_cylinder:
        querystring['min_number_of_valves_per_cylinder'] = min_number_of_valves_per_cylinder
    if min_compression_ratio:
        querystring['min_compression_ratio'] = min_compression_ratio
    if max_compression_ratio:
        querystring['max_compression_ratio'] = max_compression_ratio
    if min_kerb_weight_kg:
        querystring['min_kerb_weight_kg'] = min_kerb_weight_kg
    if max_piston_stroke_in:
        querystring['max_piston_stroke_in'] = max_piston_stroke_in
    if min_maximum_speed_mph:
        querystring['min_maximum_speed_mph'] = min_maximum_speed_mph
    if min_horsepower:
        querystring['min_horsepower'] = min_horsepower
    if max_maximum_speed_km_per_hour:
        querystring['max_maximum_speed_km_per_hour'] = max_maximum_speed_km_per_hour
    if min_acceleration_0_100_kmh_sec:
        querystring['min_acceleration_0_100_kmh_sec'] = min_acceleration_0_100_kmh_sec
    if min_cylinder_bore_in:
        querystring['min_cylinder_bore_in'] = min_cylinder_bore_in
    if max_start_of_production_year:
        querystring['max_start_of_production_year'] = max_start_of_production_year
    if max_number_of_cylinders:
        querystring['max_number_of_cylinders'] = max_number_of_cylinders
    if max_maximum_speed_mph:
        querystring['max_maximum_speed_mph'] = max_maximum_speed_mph
    if max_acceleration_0_100_kmh_sec:
        querystring['max_acceleration_0_100_kmh_sec'] = max_acceleration_0_100_kmh_sec
    if min_piston_stroke_mm:
        querystring['min_piston_stroke_mm'] = min_piston_stroke_mm
    if min_number_of_cylinders:
        querystring['min_number_of_cylinders'] = min_number_of_cylinders
    if max_horsepower:
        querystring['max_horsepower'] = max_horsepower
    if max_seats:
        querystring['max_seats'] = max_seats
    if drive_wheel:
        querystring['drive_wheel'] = drive_wheel
    if power_steering:
        querystring['power_steering'] = power_steering
    if min_end_of_production_year:
        querystring['min_end_of_production_year'] = min_end_of_production_year
    if min_start_of_production_year:
        querystring['min_start_of_production_year'] = min_start_of_production_year
    if max_wheelbase_in:
        querystring['max_wheelbase_in'] = max_wheelbase_in
    if max_wheelbase_mm:
        querystring['max_wheelbase_mm'] = max_wheelbase_mm
    if max_rear_back_track_in:
        querystring['max_rear_back_track_in'] = max_rear_back_track_in
    if max_length_mm:
        querystring['max_length_mm'] = max_length_mm
    if max_width_mm:
        querystring['max_width_mm'] = max_width_mm
    if max_height_in:
        querystring['max_height_in'] = max_height_in
    if min_wheelbase_mm:
        querystring['min_wheelbase_mm'] = min_wheelbase_mm
    if max_length_in:
        querystring['max_length_in'] = max_length_in
    if min_front_track_mm:
        querystring['min_front_track_mm'] = min_front_track_mm
    if max_rear_back_track_mm:
        querystring['max_rear_back_track_mm'] = max_rear_back_track_mm
    if min_rear_back_track_mm:
        querystring['min_rear_back_track_mm'] = min_rear_back_track_mm
    if min_rear_back_track_in:
        querystring['min_rear_back_track_in'] = min_rear_back_track_in
    if max_front_track_mm:
        querystring['max_front_track_mm'] = max_front_track_mm
    if max_max_load_lbs:
        querystring['max_max_load_lbs'] = max_max_load_lbs
    if min_max_load_kg:
        querystring['min_max_load_kg'] = min_max_load_kg
    if max_max_weight_lbs:
        querystring['max_max_weight_lbs'] = max_max_weight_lbs
    if min_doors:
        querystring['min_doors'] = min_doors
    if min_cylinder_bore_mm:
        querystring['min_cylinder_bore_mm'] = min_cylinder_bore_mm
    if min_maximum_speed_km_per_hour:
        querystring['min_maximum_speed_km_per_hour'] = min_maximum_speed_km_per_hour
    if max_doors:
        querystring['max_doors'] = max_doors
    if min_seats:
        querystring['min_seats'] = min_seats
    if max_end_of_production_year:
        querystring['max_end_of_production_year'] = max_end_of_production_year
    if assisting_systems:
        querystring['assisting_systems'] = assisting_systems
    if steering_type:
        querystring['steering_type'] = steering_type
    if fuel_type:
        querystring['fuel_type'] = fuel_type
    if model:
        querystring['model'] = model
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    if brand:
        querystring['brand'] = brand
    if title:
        querystring['title'] = title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cars-data3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

