# System Architecture

## High-Level Flow
Teradata → Alteryx → Redshift/Aurora → Analytics Dashboards

## Frontend
Secure dashboards (X-Ray) accessed via PING Federation authentication.

## Backend
Data is sourced from Teradata, transformed using Alteryx workflows, and loaded into Redshift and Aurora with incremental and versioned updates.

## Storage
Redshift supports analytics; Aurora retains historical data. Temporary tables are purged via TTL policies.
