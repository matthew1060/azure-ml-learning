# Updated training script - CI/CD test
import os
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.identity import ClientSecretCredential
import json

# Get credentials from environment
azure_credentials = os.environ.get('AZURE_CREDENTIALS', '{}')
subscription_id = os.environ.get('SUBSCRIPTION_ID')
resource_group = os.environ.get('RESOURCE_GROUP')
workspace_name = os.environ.get('WORKSPACE_NAME')
compute_name = os.environ.get('COMPUTE_NAME')

# Parse Azure credentials
try:
    creds = json.loads(azure_credentials)
    credential = ClientSecretCredential(
        tenant_id=creds.get('tenantId'),
        client_id=creds.get('clientId'),
        client_secret=creds.get('clientSecret')
    )
except:
    from azure.identity import DefaultAzureCredential
    credential = DefaultAzureCredential()

# Connect to Azure ML
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

print(f"Connected to workspace: {workspace_name}")

# Submit training job
job = command(
    display_name="CI-CD Training Run",
    code="pipeline_scripts",
    command="python train.py --input_train ${{inputs.input_train}} --output_model ${{outputs.output_model}}",
    inputs={
        "input_train": Input(type=AssetTypes.URI_FOLDER, path="azureml:iris-mltable:1")
    },
    outputs={
        "output_model": Output(type=AssetTypes.URI_FOLDER)
    },
    environment="azureml:credit-risk-environment:1",
    compute=compute_name,
    experiment_name="github-actions-training"
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Status: {returned_job.status}")