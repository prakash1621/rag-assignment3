# DOT - Complete Knowledge Base

## What is DOT?

DOT (Digital Operations Transformation) Customer Journey is a Southwest Airlines dataset that compares expected versus actual flight journeys for customers to measure experience gaps and operational performance.

## Purpose & Objectives

**Primary Goals**:
- Track expected and actual customer journeys
- Identify disruptions such as cancellations, misconnects, and diversions
- Enable analytics-driven operational improvements
- Measure customer experience gaps

**Scope**:
Journeys are captured based on:
- Cancellations (1–2 days prior)
- Midnight snapshots before departure
- First day-of-departure version

## System Architecture

### High-Level Flow
```
Teradata → Alteryx → Redshift/Aurora → Analytics Dashboards
```

**Frontend**:
- Secure dashboards (X-Ray)
- PING Federation authentication
- Role-based access control

**Backend**:
- Data sourced from Teradata
- Transformed using Alteryx workflows
- Loaded into Redshift and Aurora
- Incremental and versioned updates

**Storage**:
- **Redshift** - Supports analytics queries
- **Aurora** - Retains historical data
- Temporary tables purged via TTL policies

## Data Flow & Processing

### Authentication
PING Federation with role-based access control

### Ingestion
- Expected journey data captured via lifecycle criteria
- Combined with actual flight and passenger routing data
- CDC-based incrementals ensure freshness and traceability

### Versioning
Journey versions tracked through incremental updates for complete audit trail

## Security & Compliance

### PII Protection
- Sensitive customer data automatically detected
- Redacted before storage or analytics output
- Enforced at multiple pipeline stages

### Compliance
- Complies with Southwest privacy policies
- Data retention policies enforced
- Automated purge rules applied

## Quality Monitoring & Optimization (QMO)

### Purpose
QMO ensures data quality, security, and operational resilience of the DOT Customer Journey dataset.

### Coverage Areas
- ✅ Authentication and access control
- ✅ Schema and data validation
- ✅ Incremental load testing
- ✅ PII scrubbing verification
- ✅ Performance and resilience testing

### Certification Status
**Certified** with automation and performance validation enabled

## Testing & Validation

### Validation Strategy
- **Source-to-target reconciliation** (Teradata vs Redshift)
- **Key uniqueness** and record-count matching
- **Alteryx-based comparison** workflows
- **Specialized validation** for CUST_JRNY_TRIP_PAX_AIR_CHG

### Automation
End-to-end automated tests maintained in GitLab under DOT QMO test suite

## Core Tables

The following DOT Customer Journey tables are validated:

1. **CUST_JRNY_TRIP_FLT_LEG_CHG** - Customer journey trip flight leg changes
2. **CUST_JRNY_TRIP** - Customer journey trip data
3. **CUST_JRNY_TRIP_WTH_FLT** - Customer journey trip with flight details

## Key Features

- **Journey Comparison**: Expected vs Actual flight paths
- **Disruption Tracking**: Cancellations, misconnects, diversions
- **Version Control**: Multiple journey snapshots per trip
- **Real-time Analytics**: X-Ray dashboard access
- **Automated QA**: Continuous validation and monitoring

## Quick Reference

**What is DOT?** Customer Journey analytics system tracking expected vs actual flight experiences

**Key Tables**: CUST_JRNY_TRIP, CUST_JRNY_TRIP_FLT_LEG_CHG, CUST_JRNY_TRIP_WTH_FLT

**Architecture**: Teradata → Alteryx → Redshift/Aurora → X-Ray Dashboards

**Security**: PII protection, PING authentication, compliance-enforced

**QMO Status**: ✅ Certified with full automation

**Purpose**: Measure customer experience gaps and enable operational improvements
