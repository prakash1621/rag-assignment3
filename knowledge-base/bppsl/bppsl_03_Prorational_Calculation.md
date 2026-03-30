# Proration Calculation Logic

## Segment to Leg Proration
Segment amounts from TKT_CPN are prorated to legs using OAG_CITY_PAIR_PRRT_FCTR factors.

### Formula
Leg Amount = (Leg Proration Factor / Sum of Segment Factors) × Segment Amount

## Round Trip Handling
RT fares are split 50/50 between OB and IB before leg proration.
