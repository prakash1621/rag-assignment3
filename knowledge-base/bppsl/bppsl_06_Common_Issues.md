# Common Issues & Troubleshooting

## Missing PNRs or Flights
Check RES_PAX_AIR load timing delays.

## Incorrect Dates or Itinerary Types
Usually caused by delayed RES_PAX_AIR CDC processing.

## Incorrect Fare or Amounts
Validate RES_TKT_ELEM_CPN load timing and eff_to_cent_tz logic.

## Missing Leg Information
Verify SCHD_FLT_LEG has active published schedules.
