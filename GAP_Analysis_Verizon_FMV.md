# GAP Analysis: Verizon FMV Design vs. Implementation

**Document:** Comparison of Verizon_FMV_HLD.docx, Verizon_FMV_LLD.docx, and `06_census_data_pipeline.ipynb`  
**Date:** February 2026  
**Scope:** Data pipeline, feature engineering, ML pipeline, and infrastructure

---

## Executive Summary

The **Census Data Pipeline** notebook (06) provides a strong foundation for **geographic and demographic data** at multiple Census hierarchy levels. It aligns well with the HLD/LLD's **Domain 5 (Microeconomic)** and partially with **Domain 6 (Macro)** and **Domain 7 (Competition)**. However, significant gaps remain in:

1. **ML pipeline** — No clustering, pseudo-labeling, supervised model, or forecasting
2. **Data sources** — Internal lease data, FCC ASR, county land values, and zoning not integrated
3. **Feature completeness** — Several LLD-specified features require site-level or external data not yet joined
4. **Infrastructure** — No API, dashboard, orchestration, or simulation suite

---

## 1. Architecture & Pipeline Gaps

| HLD/LLD Component | Specified | Implemented in NB06 | Gap |
|------------------|-----------|---------------------|-----|
| **6-Stage ML Pipeline** | Clustering → Pseudo-label → Supervised → Classification → Validation → Forecasting | None | **Full gap** — NB06 is data collection only |
| **10-Step Data Prep** | extract_lease → fetch_census → fetch_fcc → fetch_land → geospatial_join → cleaning → feature_engineering → run_pipeline → run_validation → run_forecasting | Steps 2 (Census), 5 (partial geospatial) | Steps 1, 3, 4, 6–10 not implemented |
| **Project Structure** | `verizon_fmv/src/ingestion/`, `processing/`, `models/`, `serving/`, etc. | Notebooks + `scripts/ingest_data.py` | No `src/` module structure per LLD |
| **Orchestration** | Apache Airflow DAG | None | No orchestration |
| **API** | FastAPI `/api/v1/predict-fmv`, `/forecast`, `/portfolio/summary` | None | No serving layer |
| **Dashboard** | Streamlit (Portfolio Overview, Site Drill-Down, Renegotiation Priority, etc.) | None | No consumption layer |

---

## 2. Data Sources Gaps

### 2.1 HLD Data Domains vs. Implementation

| Domain | HLD Source | NB06 | NB07 | NB08 | Gap |
|--------|------------|------|------|------|-----|
| **D1** Internal Contract & Payments | Verizon Lease DB | ❌ | ❌ | ❌ | **Full gap** — Internal; no integration |
| **D2** Zoning & Regulatory | Municipal GIS, Wharton WRLURI | ❌ | ❌ | ✅ WRLURI | NB06/07 lack zoning; NB08 has WRLURI |
| **D3** Physical Cell Site | FCC ASR, site surveys | ❌ | ❌ | ❌ | **Full gap** — OpenCelliD used as tower proxy, not FCC ASR |
| **D4** County Land Values | County assessor, USDA | ❌ | ❌ | Partial (USDA in NB08) | Not in NB06 |
| **D5** Microeconomic | Census ACS | ✅ Full | Partial | — | **Aligned** |
| **D6** Macro | CPI, GDP, unemployment | ✅ FRED, BLS LAUS | — | ✅ BEA GDP, FRED | NB06 has FRED + BLS; GDP per MSA in NB08 |
| **D7** Competition | FCC Tower Data | Partial (OpenCelliD) | Partial | — | FCC ASR not used; OpenCelliD is proxy |

### 2.2 Geospatial Enrichment Joins (HLD: 14 total)

| Join | HLD Spec | NB06 | Gap |
|------|----------|------|-----|
| Tract demographics | Census ACS | ✅ | None |
| Tract area (population density) | TIGER | ✅ | None |
| FHFA HPI | State/ZIP | ✅ | None |
| OpenCelliD towers | Tract aggregation | ✅ | None |
| FCC ASR tower count 5km | Site radius | ❌ | **Gap** — FCC ASR not fetched |
| FCC tower density | Tract | ❌ | **Gap** |
| Competitor tower distance | Nearest-neighbor | ❌ | **Gap** |
| USGS elevation | Site | ❌ | NB07 | NB06 does not join |
| FEMA flood | Site | ❌ | NB07 | NB06 does not join |
| TIGER highway distance | Site | ❌ | NB07 | NB06 does not join |
| HUD SAFMR | ZIP | ❌ | NB08 | NB06 does not join |
| Urban/rural class | Tract | ❌ | NB07 | NB06 does not join |
| WRLURI / permit difficulty | State/County | ❌ | NB08 | NB06 does not join |
| County land values | County | ❌ | — | **Gap** |

---

## 3. Feature Engineering Gaps

### 3.1 LLD Tier 1 (X_market ~45 features) vs. NB06

| LLD Feature | Domain | NB06 Raw/Derived | Gap |
|-------------|--------|------------------|-----|
| `census_population_density_log` | D5 | ✅ `population_density` (tract) | Need log transform |
| `census_population_3km_log` | D5 | ❌ | **Gap** — Requires site-level 3km buffer; tract-only in NB06 |
| `census_median_income_scaled` | D5 | ✅ `median_income` | Need scaling |
| `census_median_home_value_log` | D5 | ✅ `median_home_value` | Need log transform |
| `census_median_rent_log` | D5 | ✅ `median_rent` | Need log transform |
| `census_vacancy_rate` | D5 | ✅ `vacancy_rate` | **Aligned** |
| `urban_rural_class_ordinal` | D5 | ❌ | NB07 | NB06 does not compute |
| `underlying_land_value_psf_log` | D5 | ❌ | NB08 (HUD proxy) | NB06 does not compute |
| `hud_safmr_log` | D5 | ❌ | NB08 | NB06 does not fetch |
| `property_tax_rate` | D5 | ❌ | — | **Gap** — Not in any notebook |
| `topography_class_ordinal` | D5 | ❌ | NB07 | NB06 does not compute |
| `terrain_elevation_variance_m` | D5 | ❌ | NB07 | NB06 does not compute |
| `poi_density_1km` | D5 | ❌ | — | **Gap** — External (OSM/Google) |
| `building_density_500m` | D5 | ❌ | — | **Gap** |
| `unemployment_rate_local` | D6 | ✅ BLS LAUS (county) | **Aligned** (BLS optional) |
| `gdp_per_capita_msa_scaled` | D6 | ❌ | NB08 BEA | NB06 does not fetch |
| `fcc_tower_count_5km` | D7 | ❌ | — | **Gap** — FCC ASR not used |
| `fcc_tower_density_sqkm` | D7 | ❌ | — | **Gap** |
| `competitor_tower_distance_km` | D7 | ❌ | — | **Gap** |
| `tower_per_capita` | D7 | Partial (OpenCelliD towers / population) | OpenCelliD used, not FCC |
| `scarcity_index` | D7 | ❌ | NB07 composite | NB06 does not compute |
| `composite_hazard_score` | D7 | ❌ | NB07 | NB06 does not compute |
| `distance_to_highway_km` | D7 | ❌ | NB07 | NB06 does not compute |
| `ground_elevation_ft` | D7 | ❌ | NB07 | NB06 does not compute |
| `landlord_concentration` | D7 | ❌ | Lease data | **Gap** — Requires internal data |

### 3.2 NB06 Features Present (28 raw + 8 derived)

**Raw:** population, median_income, median_home_value, median_rent, median_contract_rent, total_housing_units, vacant_housing_units, owner_occupied_units, renter_occupied_units, median_year_built, rent_burden_pct, median_monthly_housing_cost, median_age, gini_index, poverty_population, total_commuters, transit_commuters, total_workers_commute, commute_60_89_min, commute_90_plus_min, bachelors_degree, masters_degree, doctorate_degree, in_labor_force, unemployed_civilian, employed_agriculture_mining, employed_construction, employed_professional_scientific  

**Derived:** vacancy_rate, owner_renter_ratio, transit_commuter_pct, long_commute_pct, college_degree_pct, labor_force_participation, poverty_rate, professional_employment_pct  

**Additional in NB06:** population_density (tract), temporal growth rates (1yr/3yr/5yr), hierarchical imputation

### 3.3 LLD Tier 2 (Z_network ~19 features)

All Tier 2 features require **internal Verizon data** (tenant_count, coverage_critical, backhaul_type, etc.). NB06 does not and cannot provide these — **expected gap** until lease/network data is integrated.

### 3.4 LLD Tier 4 (Interaction Features)

| Interaction | Components | NB06 | Gap |
|-------------|------------|------|-----|
| `income_density_interaction` | log(median_income × population_density) | Can compute from NB06 | Minor — add in feature layer |
| `tower_competition_density` | fcc_tower_count_5km / population_density | ❌ | **Gap** — Needs FCC ASR |
| `scarcity_x_demand` | scarcity_index × census_population_3km | ❌ | **Gap** — Needs both |
| `permit_difficulty_x_density` | permit_approval_difficulty × density | ❌ | NB08 × NB06 | Cross-notebook join needed |
| `hpi_x_scarcity` | hpi_appreciation_3yr × scarcity_index | ❌ | NB08 × NB07 | Cross-notebook join needed |

---

## 4. Geographic Hierarchy Alignment

| Level | HLD/LLD | NB06 | Status |
|-------|---------|------|--------|
| National | FRED macro | ✅ CPIAUCSL, DGS10, UNRATE, HOUST | **Aligned** |
| State | ACS + FRED state unemployment | ✅ | **Aligned** |
| County | ACS + BLS LAUS | ✅ (BLS optional) | **Aligned** |
| Tract | ACS + TIGER + FHFA HPI + OpenCelliD | ✅ | **Aligned** |
| Block Group | ACS (finest) | ✅ (latest year only) | **Aligned** |
| ZCTA/ZIP | ACS + Gazetteer + FHFA HPI | ✅ | **Aligned** |
| **Site** (lat/lon) | Required for FMV prediction | ❌ | **Gap** — NB06 is geography-level; site join happens elsewhere |

**Year range:** 2015–2024 (10 vintages) — **Aligned** with HLD timeline.

---

## 5. Model Configuration Alignment

`configs/model_config.yaml` aligns with LLD:

| Parameter | LLD | model_config.yaml | Status |
|-----------|-----|-------------------|--------|
| FMV coefficients | log_median_home_value 0.35, scarcity_index 0.15, etc. | ✅ Matches | **Aligned** |
| HDBSCAN min_cluster_size | 20 | 20 | **Aligned** |
| MAD threshold | 1.0 | 1.0 | **Aligned** |
| QRF quantiles | [0.10, 0.25, 0.50, 0.75, 0.90] | ✅ | **Aligned** |
| Over/Under bounds | Q75 / Q25 | q75 / q25 | **Aligned** |

---

## 6. Gaps Summary by Priority

### Critical (Blocking FMV Model)

1. **Verizon Lease DB integration** — No current_rent, lease terms, or site identifiers
2. **ML pipeline implementation** — Stages 1–6 (clustering through forecasting) not built
3. **Site-level feature join** — NB06 outputs tract/county/state; FMV needs site (lat/lon) → tract/county join
4. **FCC ASR tower data** — Competition features (fcc_tower_count_5km, density, competitor distance) require FCC, not just OpenCelliD

### High (Reduces Model Quality)

5. **census_population_3km** — Site-level 3km buffer aggregation
6. **gdp_per_capita_msa** — BEA/FRED MSA GDP (NB08 references; not in NB06)
7. **Unified feature matrix** — Single table joining NB06 + NB07 + NB08 for model input
8. **Log/scaling transforms** — Apply log() and scaling per LLD spec

### Medium (Enhancement)

9. **HUD SAFMR join** — NB08 has it; needs to be joined to tract/ZIP in unified pipeline
10. **WRLURI / permit_difficulty** — NB08 has it; join to state/county
11. **Property tax rate** — Not in any notebook
12. **POI density, building density** — External sources

### Low (Infrastructure)

13. **API (FastAPI)** — Serving layer
14. **Dashboard (Streamlit)** — Consumption layer
15. **Airflow orchestration** — DAG for 10-step pipeline
16. **Simulation suite** — sim_00 through sim_07
17. **Testing** — pytest, Great Expectations

---

## 7. Recommendations

1. **Short term:** Build a **unified feature matrix** script that joins NB06 (Census), NB07 (geospatial), and NB08 (housing/regulatory) outputs on tract_fips, county_fips, state_abbr, zcta. Add log and scaling transforms per LLD.
2. **Medium term:** Implement **FCC ASR** ingestion and radius queries for `fcc_tower_count_5km`, `fcc_tower_density_sqkm`, and `competitor_tower_distance_km`. Integrate BEA/FRED MSA GDP for `gdp_per_capita_msa_scaled`.
3. **Long term:** Implement the full **6-stage ML pipeline** (clustering → pseudo-label → supervised → classification → validation → forecasting) once Verizon lease data is available. Add API, dashboard, and orchestration per LLD.
4. **Documentation:** Maintain a feature-to-source mapping (as in NB08 Section 12) and keep it updated as new data sources are added.

---

## Appendix: Cross-Reference

| Document | Section | Content |
|----------|---------|---------|
| HLD | §5 Data Sources | 7 domains, 14 geospatial joins |
| HLD | §7 ML Pipeline | 6 stages |
| LLD | §8 Feature Engineering | Tier 1–4, ~75–80 features |
| LLD | §9 Data Pipeline | 11-step DAG |
| NB06 | Header | Geographic hierarchy, 28+8 features |
| NB08 | §12 | Complete FMV feature map (NB06/07/08 → LLD) |
