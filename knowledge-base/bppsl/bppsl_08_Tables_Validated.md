# BPPSL Tables Validated

The following BPPSL (Booked Passenger Proration Segment/Leg) tables were validated as part of the BPPSL migration:

- BKNG_PNR_PAX_SEG_LEG
- ITIN_FLT_PATH
- TEMP_LOAD_DEFAULT_FARE
- DEFAULT_FARE

These are the core BPPSL dataset tables for booking and fare proration.


1. BKNG_PNR_PAX_SEG_LEG
What it is:
The main BPPSL fact table in EDW that holds booking data at the PNR / Passenger / Segment / Leg level.

Purpose / usage:

Represents each passenger’s booked travel broken down to individual flight legs.

Stores itinerary structure (path, part, leg types), booking method/source, fare basis, fare source, passenger type, revenue amounts, etc.

Acts as the primary source for downstream analytics, including default fare calculations.


2. ITIN_FLT_PATH
What it is:
An itinerary flight path dimension in CDW/EDW that describes the origin–destination path for an itinerary (O&D), independent of individual legs.

Purpose / usage:

Stores unique itinerary paths: origin and destination airports for a given flight path.

Joined from BPPSL via MKT_OPNG_ITIN_FLT_PATH_ID to get market-level origin/dest.

From the tech docs:

Purpose: “Itinerary flight path dimension table”

Key fields: ITIN_FLT_PATH_ID, ITIN_ORIG_ARPT_CDE, ITIN_DEST_ARPT_CDE

3. TEMP_LOAD_DEFAULT_FARE
What it is:
A session-scoped temporary table created inside sp_default_fare_load to stage aggregated default fare stats.

Structure (from docs):

MKT_ORIG – market origin airport (from ITIN_ORIG_ARPT_CDE)

MKT_DEST – market destination airport (from ITIN_DEST_ARPT_CDE)

DEP_DATE – market departure date (MKT_DEP_DT)

FARE_BASIS_CD – fare basis (FARE_BSIS_ID)

CONTRIB_BOOKING_COUNT – sum of contributing bookings

CONTRIB_REVENUE_AMOUNT – sum of contributing revenue