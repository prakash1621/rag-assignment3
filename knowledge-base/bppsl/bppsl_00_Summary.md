# BPPSL - Complete Knowledge Base

## What is BPPSL?

BPPSL (Booked Passenger Proration Segment/Leg) is a Southwest Airlines data system that breaks down booked itineraries (PNR/Passenger) to segment and leg level for revenue-impacting bookings that remove seats from inventory.

## Why is BPPSL Critical ?

Over time BPPSL has been leveraged for analytics by a larger audience within the organization. Since it combines the itinerary and revenue data as a snapshot for the end of the day, it minimizes the need to join multiple tables and cleanse the data making it a single source of reference. Revenue Management, Network Planning, Southwest Business, Marketing, Finance, FP&A and the Executive Offices are some of the major Business and Technology Consumers of this data. This is one of the key source data for Revenue Forecast tools, Bid Price Translation, Rodeo, RMS, Benchmark skycast accuracy and scalpel tool to modify the Flight Schedule. The data is used to pull reports for a large number of Senior Leaders, including Gary Kelly, and is leveraged for day to day business decisions. It really helps in bettering revenue forecasts and schedule optimizations.


## Purpose

BPPSL provides detailed booking and fare allocation data at the segment and leg level, enabling accurate revenue tracking and analysis.

## Scope

**In Scope**:
- Revenue and seat-impacting bookings
- Segment and leg-level breakdown
- Prorated fare allocation

**Out of Scope**:
- Lap infants
- Same-day CDCs (only latest activity per day)
- Same-day cancelled or refunded PNRs

## Fare Source Logic

### Order of Precedence
**TC → TK → TT → LO → GP → RF → DG → GF → Defaults**

**Key Rules**:
- **TC** - Coupon-level proration (highest priority)
- **TK/TT** - Used when TC is unavailable
- **LO** - Loyalty bookings (base fare only)
- **RF** - Unticketed bookings with SBR pricing
- **GP/DG** - Group bookings
- **GF** - Government fares
- **Defaults** - Used when no criteria is met

## Proration Calculation

### Segment to Leg Proration
Segment amounts from TKT_CPN are prorated to legs using OAG_CITY_PAIR_PRRT_FCTR factors.

**Formula**:
```
Leg Amount = (Leg Proration Factor / Sum of Segment Factors) × Segment Amount
```

### Round Trip Handling
RT fares are split 50/50 between OB (outbound) and IB (inbound) before leg proration.

## Special Fare Types

### Government Fare (GF)
- Applied when PNR qualifies as GOV and no higher-precedence fare exists
- Uses govt_fare table with city pair, booking class, and journey date
- Supports fallback logic (YCA → YDG → blended fares)
- Applicable only to non-stop itineraries

### Group Fare (DG)
- Applies when PNR_TYPE = GRP without NETFARE
- Base fare derived from first passenger in RES_FARE
- Same amount applied across group passengers

## Data Quality

**Snapshot Characteristics**:
- BPPSL is a prior-day snapshot
- Limited revenue fluctuation unless PNRs cancel/refund

**Load Dependencies**:
- All base tables must load before BPPSL run
- Metrics support correct proration filtering

## Common Issues & Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Missing PNRs or Flights | RES_PAX_AIR load timing delays | Check load timing |
| Incorrect Dates/Itinerary Types | Delayed RES_PAX_AIR CDC processing | Verify CDC processing |
| Incorrect Fare/Amounts | RES_TKT_ELEM_CPN load timing issues | Validate load timing and eff_to_cent_tz logic |
| Missing Leg Information | Missing published schedules | Verify SCHD_FLT_LEG has active schedules |

## Recent Changes

**Tenasi Changes**:
- Source IDW tables moved from feeds to events
- No BPPSL logic changes required

**Galaxy Fix**:
- Marketing carrier now sourced correctly from RPA.mktg_carr_cde

## Tables Validated

Core BPPSL dataset tables:
1. **BKNG_PNR_PAX_SEG_LEG** - Main booking proration table
2. **ITIN_FLT_PATH** - Itinerary flight path data
3. **TEMP_LOAD_DEFAULT_FARE** - Temporary default fare staging
4. **DEFAULT_FARE** - Default fare reference table

## Quick Reference

**What is BPPSL?** Booking proration system that breaks down PNR data to segment/leg level

**Key Tables**: BKNG_PNR_PAX_SEG_LEG, ITIN_FLT_PATH, DEFAULT_FARE

**Fare Precedence**: TC > TK > TT > LO > GP > RF > DG > GF > Defaults

**Proration**: Segment amounts split to legs using OAG proration factors

**Special Fares**: Government (GF) and Group (DG) fares have specific logic
