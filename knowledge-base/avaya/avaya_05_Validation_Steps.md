# Validation Steps

## Data Preparation
Sample Parquet files were manually placed in QA curated S3 buckets to simulate upstream ingestion.

Sample Locations:
- s3://datalake-us-east-1-qa1-curated-secure/customer/ech/external-call-history/
- s3://datalake-us-east-1-qa1-curated-secure/customer/cms/agent-trace/
- s3://datalake-us-east-1-qa1-curated-secure/customer/avaya-sms/optin/

## Trigger Verification
- Confirmed Lambda triggers executed successfully
- CloudWatch logs reviewed

## Glue Job Verification
- Glue jobs executed for all three Avaya datasets
- Logs reviewed for errors

## Data Validation
- Verified row counts and schema alignment
- Confirmed data availability in target views
