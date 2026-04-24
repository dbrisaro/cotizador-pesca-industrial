"""
How the cotizador CSVs were built
==================================
Source pipeline: /home/jupyter-daniela/peru_catch_modeling/scripts/pipeline/

-----------------------------------------------------------------------
cotizador_company_baselines.csv
-----------------------------------------------------------------------
Script:  step18_client_company_analysis.py

For each company and each season type (T1 / T2):
  baseline = mean annual catch in tons across all years that company
             reported to IHMA (coverage: 2015-2025, but not every
             company has every year).

  T1 = primera temporada (DOY 91-212, approx Apr-Jul)
  T2 = segunda temporada (DOY 305-365, approx Nov-Dec)

Source data: calas_enriched.csv (output of step05_data_enrichment.py),
  which aggregates raw IHMA cala-level records to company x year x season.

-----------------------------------------------------------------------
cotizador_sst_by_season.csv
-----------------------------------------------------------------------
Scripts: step04b_sst_modis.py  (daily anomaly files)
         step13_bootstrap_aep.py (seasonal aggregation)

For each year (2002-2025) and each season type:
  sst = mean daily MODIS AQUA SST anomaly over the season DOY range,
        spatially averaged over the Centro Norte fishing corridor
        (lat -11.0 to -7.1 S, clipped to the observed fishing polygon
        defined by the 5th-95th percentile of cala longitudes per
        1-degree latitude band).

Anomaly definition:
  anomaly(day) = raw_SST(day) - climatology(DOY)
  climatology  = mean raw SST for that day-of-year over 2005-2024
                 (20-year reference period)

Season DOY ranges:
  T1: DOY 91-212  (April 1 - July 31)
  T2: DOY 305-365 (November 1 - December 31)

-----------------------------------------------------------------------
cotizador_company_actuals.csv
-----------------------------------------------------------------------
Script:  step18_client_company_analysis.py

For each company x year x season:
  actual_ton = total catch in tons declared to IHMA that season.

Raw source: IHMA CSV files in
  /home/jupyter-daniela/suyana/peru_production/inputs/ihma_data/{year}/
  pattern: *Calas*anchoveta*{Primera|Segunda}temporada*.csv

Gaps in the actuals:
  - 2015-T1: no data (veda or no IHMA records)
  - 2024-T1: no data (veda or no IHMA records)
  - Pre-2015: no company-level data available

-----------------------------------------------------------------------
OLS reference curve (shown in chart)
-----------------------------------------------------------------------
Script:  step15_trigger_design.py

  loss_fraction = 1 - exp(beta * SST_anomaly)   when SST_anomaly > 0
  beta = -0.816   (OLS M1, empresa x temporada, Centro Norte)

This curve was estimated by regressing observed catch losses
(baseline - actual) / baseline against seasonal SST anomaly,
across all company x season observations in 2015-2025.
"""
