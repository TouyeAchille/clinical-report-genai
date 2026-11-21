# Étape 1 – Caractérisation de PARHAF

Patients : 4254 | Documents : 6185 | Coût fixe du prompt : 89 tokens | Tokens par mot : 2.57

## Par type de document
| type         |    n |   mots_median |   tokens_median |   tokens_p90 |   tokens_max |   pct_tient_256 |   pct_tient_512 |
|:-------------|-----:|--------------:|----------------:|-------------:|-------------:|----------------:|----------------:|
| CRH          | 3563 |         496   |          1278   |       2574   |         7889 |            70.4 |            59.7 |
| CRC          | 1079 |         232   |           534   |        989.4 |         7090 |            98.8 |            96.8 |
| CRO          |  889 |         220   |           557   |       1008.4 |         2629 |            97.6 |            96.2 |
| ANAPATH      |  433 |         261   |           739   |       1463.6 |         3065 |            93.3 |            89.6 |
| ACCOUCHEMENT |   74 |         201   |           515.5 |       1178.1 |         1367 |           100   |           100   |
| MATERNITE    |   74 |         117.5 |           321.5 |        548.4 |          620 |           100   |           100   |
| URGENCES     |   73 |         345   |           905   |       1315.2 |         1604 |           100   |            95.9 |

## Patients dont tous les documents tiennent dans la fenêtre (%)
- sortie réservée 256 : 64.3 %
- sortie réservée 512 : 52.0 %

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