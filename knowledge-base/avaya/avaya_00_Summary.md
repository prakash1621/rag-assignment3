# Avaya - Complete Knowledge Base

## What is Avaya?

Avaya is a customer communication and call center data processing module at Southwest Airlines. It handles SMS opt-in data, agent call traces, and external call history for customer analytics.

## Migration Project Overview

**Purpose**: Migrate Avaya data processing module from AWS CloudFormation to Terraform

**Business Value**:
- Standardizes infrastructure-as-code using Terraform
- Improves maintainability and scalability of Avaya pipelines
- Ensures uninterrupted data availability for customer analytics

## Scope

**In Scope**:
- Migration of Avaya module from CloudFormation to Terraform
- Validation of Lambda triggers
- Validation of AWS Glue jobs
- Data verification for Avaya tables

**Out of Scope**:
- Non-Avaya pipelines
- Production cutover activities

## Architecture & Infrastructure

**Infrastructure-as-Code**:
- Source: AWS CloudFormation
- Target: Terraform

**S3 Script Locations**:

CloudFormation:
- `datalake-us-east-1-dev1-glue-job-scripts/decp/customer/avaya/glue-python-scripts/`
- `datalake-us-east-1-dev1-glue-job-scripts/decp/customer/avaya/SQL/`

Terraform:
- `datalake-us-east-1-dev1-glue-job-scripts/decp/customer/avaya/glue-scripts/`
- `datalake-us-east-1-dev1-glue-job-scripts/decp/customer/avaya/sql-scripts/`

## Tables Validated

The following Avaya tables were validated:

1. **customer_vw.avaya_cba_sms_optin** - SMS opt-in data
2. **customer_vw.avaya_cms_agent_trace** - Agent call traces
3. **customer_vw.avaya_ech_external_call_history** - External call history

## Validation Process

### Data Preparation
Sample Parquet files were manually placed in QA curated S3 buckets:
- `s3://datalake-us-east-1-qa1-curated-secure/customer/ech/external-call-history/`
- `s3://datalake-us-east-1-qa1-curated-secure/customer/cms/agent-trace/`
- `s3://datalake-us-east-1-qa1-curated-secure/customer/avaya-sms/optin/`

### Trigger Verification
- Lambda triggers executed successfully
- CloudWatch logs reviewed

### Glue Job Verification
- Glue jobs executed for all three Avaya datasets
- Logs reviewed for errors

### Data Validation
- Verified row counts and schema alignment
- Confirmed data availability in target views

## Results

**Validation Outcome**:
- ✅ Terraform-based deployment completed successfully
- ✅ Lambda and Glue orchestration functioned as expected
- ✅ All validated tables populated with correct data
- ✅ No issues observed during QA validation

## Conclusion

The Avaya module migration from CloudFormation to Terraform was successfully validated in QA. All critical components including infrastructure, triggers, data processing jobs, and output tables performed as expected, confirming readiness for higher environment promotion.

## Quick Reference

**What is Avaya?** Customer communication and call center data module

**Key Tables**: 
- avaya_cba_sms_optin
- avaya_cms_agent_trace  
- avaya_ech_external_call_history

**Migration**: CloudFormation → Terraform ✅ Successful

**Status**: QA Validated, Ready for Production
