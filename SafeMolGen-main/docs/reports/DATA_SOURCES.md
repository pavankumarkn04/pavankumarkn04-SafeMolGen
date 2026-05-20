# Data Sources (Only What We Use)

This file lists only the datasets and database exports that we will use in
this project. If we add or remove a source later, update this list.

## ChEMBL (chembl_36)
Purpose: Phase 3 generator pretraining (SMILES corpus).
Planned file: `chembl_36_chemreps.txt.gz` (SMILES/representations with ChEMBL IDs).
Attribution: ChEMBL data is from http://www.ebi.ac.uk/chembl - version chembl_36.
License: CC Attribution-ShareAlike 3.0 Unported.

Required citation for publications:
Mendez D, Gaulton A, Bento AP, Chambers J, De Veij M, Felix E, Magarinos MP,
Mosquera JF, Mutowo P, Nowotka M, Gordillo-Maranon M, Hunter F, Junco L,
Mugumbate G, Rodriguez-Lopez M, Atkinson F, Bosc N, Radoux CJ, Segura-Cabrera A,
Hersey A, Leach AR. ChEMBL: towards direct deposition of bioassay data.
Nucleic Acids Res. 2019 47(D1):D930-D940. DOI: 10.1093/nar/gky1075

Links:
- https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/README
- https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/REQUIRED.ATTRIBUTION

## TDC ADMET Benchmarks
Purpose: Phase 1 ADMET predictor training and evaluation.
Planned usage: Use TDC ADMET Group (22 endpoints) through the `tdc` Python library.
Endpoints:
- Caco2_Wang, HIA_Hou, Pgp_Broccatelli, Bioavailability_Ma,
  Lipophilicity_AstraZeneca, Solubility_AqSolDB,
  BBB_Martins, PPBR_AZ, VDss_Lombardo,
  CYP2C9_Veith, CYP2D6_Veith, CYP3A4_Veith,
  CYP2C9_Substrate_CarbonMangels, CYP2D6_Substrate_CarbonMangels,
  CYP3A4_Substrate_CarbonMangels, Half_Life_Obach,
  Clearance_Hepatocyte_AZ, Clearance_Microsome_AZ,
  LD50_Zhu, hERG, AMES, DILI.
Attribution: follow TDC dataset citation guidance (add paper/URL on final writeup).
Link: https://tdcommons.ai/benchmark/admet_group/overview/

## TrialBench / Clinical Trial Outcomes
Purpose: Phase 2 DrugOracle training (clinical phase predictors).
Planned usage: Use the TrialBench clinical trial dataset (source to be fixed).
Attribution: to be added when dataset is fixed.
