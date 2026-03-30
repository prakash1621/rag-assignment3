# Government (GVT) & Group (GRM) Fare Logic

## Government Fare (GF)
- Applied when PNR qualifies as GOV and no higher-precedence fare exists
- Uses govt_fare table with city pair, booking class, and journey date
- Supports fallback logic (YCA → YDG → blended fares)
- Applicable only to non-stop itineraries

## Group Fare (DG)
- Applies when PNR_TYPE = GRP without NETFARE
- Base fare derived from first passenger in RES_FARE
- Same amount applied across group passengers
