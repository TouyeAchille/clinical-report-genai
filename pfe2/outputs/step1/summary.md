# Étape 1 – Caractérisation de PARHAF

Patients : 4254 | Documents : 6185 | Coût fixe du prompt : 177 tokens | Tokens par mot : 2.57

## Par type de document
| type         |    n |   mots_median |   tokens_median |   tokens_p90 |   tokens_max |   pct_tient_256 |   pct_tient_512 |
|:-------------|-----:|--------------:|----------------:|-------------:|-------------:|----------------:|----------------:|
| CRH          | 3563 |         496   |          1278   |       2574   |         7889 |            67.5 |            55.2 |
| CRC          | 1079 |         232   |           534   |        989.4 |         7090 |            98.1 |            95.7 |
| CRO          |  889 |         220   |           557   |       1008.4 |         2629 |            97   |            95.6 |
| ANAPATH      |  433 |         261   |           739   |       1463.6 |         3065 |            92.4 |            86.8 |
| ACCOUCHEMENT |   74 |         201   |           515.5 |       1178.1 |         1367 |           100   |            98.6 |
| MATERNITE    |   74 |         117.5 |           321.5 |        548.4 |          620 |           100   |           100   |
| URGENCES     |   73 |         345   |           905   |       1315.2 |         1604 |           100   |            90.4 |

## Patients dont tous les documents tiennent dans la fenêtre (%)
- sortie réservée 256 : 60.7 %
- sortie réservée 512 : 46.7 %

## Remplissage de structured_abstract (%)
|                 |    0 |
|:----------------|-----:|
| abs_diagnostic  | 85   |
| abs_acte        | 24.2 |
| abs_mode_entree | 37.6 |
| abs_mode_sortie | 37.6 |
| abs_duree       | 83.1 |

## Combinaisons de documents par patient
| doc_types                       |   count |
|:--------------------------------|--------:|
| CRH                             |    2672 |
| CRC+CRH+CRO                     |     887 |
| ANAPATH                         |     421 |
| CRC                             |     192 |
| ACCOUCHEMENT+MATERNITE+URGENCES |      72 |
| ANAPATH+ANAPATH                 |       6 |
| ACCOUCHEMENT+CRH+MATERNITE      |       2 |
| CRH+CRO                         |       1 |
| CRH+CRO+URGENCES                |       1 |