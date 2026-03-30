# Data Quality Considerations

## Snapshot Characteristics
- BPPSL is a prior-day snapshot
- Limited revenue fluctuation unless PNRs cancel/refund

## Load Dependencies
- All base tables must load before BPPSL run
- Metrics support correct proration filtering
