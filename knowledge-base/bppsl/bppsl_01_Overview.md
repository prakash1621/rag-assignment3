# BPPSL Overview

## Introduction
BPPSL breaks down booked itineraries (PNR/Passenger) to segment and leg level for revenue-impacting bookings that remove seats from inventory.


## BUSINESS PROBLEM
Improve Technology foundation for BPPSL reporting from a stability, timely and reliability perspective. Service Level Agreement (SLA) are currently not being met pretty frequently for critical downstream business process and decision making. Inability to eliminate SLA issues for downstream consumers with ageing technology (Teradata) and reliance on daily batch processing. Default fare records not within acceptable limits indicating data incompleteness. Limited monitoring, time consuming restart and recovery, absence of reconciliation to check on data quality.  



## BUSINESS VALUE
Implementing the new BPPSL in phases for on-prem and cloud is an attempt to bring back stability and reliability to the product. Improve SLA metrics by moving to Sabre-IX proration and CDS-A feeds as source for BPPSL. Improve inbuilt data validation checks and balances before deeming BPPSL run complete. Potentially Reduce Default fare percentage relying on CDS-A alone as source. The cloud solution will enhance platform capabilities, performance, monitoring, improved recovery and restarts. Align with technology modernization


## In Scope
- Revenue and seat-impacting bookings
- Segment and leg-level breakdown
- Prorated fare allocation

## Out of Scope
- Lap infants
- Same-day CDCs (only latest activity per day)
- Same-day cancelled or refunded PNRs
