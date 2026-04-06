# Third-Party Data and Attribution Notices

This repository includes or derives from external data sources and routing services.
Those materials are not claimed as original repository-authored content and remain
subject to their own attribution and license terms.

## Hamburg Geoportal / Transparenzportal

### Vollstationaere Pflegeeinrichtungen Hamburg

- Source:
  `https://suche.transparenz.hamburg.de/dataset/vollstationaere-pflegeeinrichtungen-hamburg13`
- Access service:
  `https://geodienste.hamburg.de/HH_WFS_Vollstationaere_Pflegeeinrichtungen?SERVICE=WFS&REQUEST=GetCapabilities`
- License:
  `Datenlizenz Deutschland Namensnennung 2.0`
- Required attribution:
  `Freie und Hansestadt Hamburg, Behoerde fuer Arbeit, Gesundheit, Soziales, Familie und Integration`
- Repository materials:
  `app/backend/data/de_hh_up_vollstationaere_pflegeeinrichtungen_EPSG_4326.json`
  `app/backend/data/de_hh_up_vollstationaere_pflegeeinrichtungen_EPSG_4326-Copy1.json`
  and benchmark inputs derived from these source locations and capacities.

### Notunterkuenfte Hamburg

- Source:
  `https://geodienste.hamburg.de/HH_WFS_Notunterkuenfte?SERVICE=WFS&REQUEST=GetCapabilities`
- License:
  `Datenlizenz Deutschland Namensnennung 2.0`
- Required attribution:
  `Freie und Hansestadt Hamburg, Bezirksamt Hamburg-Mitte`
- Repository materials:
  benchmark problem instances and derived benchmark inputs containing Hamburg shelter / depot locations.

## openrouteservice

- Terms of service:
  `https://openrouteservice.org/terms-of-service/`
- Results license:
  `CC-BY 4.0`
- Attribution:
  `(c) openrouteservice.org by HeiGIT | Map data (c) OpenStreetMap contributors`
- Repository materials:
  `benchmark_data/precomputed_matrices/matrices/*.json`
  `benchmark_data/solutions/**/matrices/*.json`
  and related first-leg routing artifacts generated with openrouteservice.

## Modification Notice

Benchmark instances in this repository are transformed and repackaged for research use.
When redistributing derived benchmark data, preserve the original source attributions
above and indicate that the source data were transformed for benchmark use.
