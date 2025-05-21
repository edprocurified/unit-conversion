import json
import logging
import os
from pathlib import Path
import sys

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# S3 Configuration
S3_BUCKET = "qcv3-us-east-2"  # Default bucket name
AWS_PROFILE = "rupert-tait"    # Default SSO profile
DATA_DIR = "data"              # Local data directory

# Quote UUID to download (change this value to the quote you want)
QUOTE_UUID = "1747666193702x771505423760365200"

def init_s3_client(profile=AWS_PROFILE):
    """Initialize and return an S3 client using boto3."""
    try:
        # Try to create a boto3 session with the SSO profile
        session = boto3.Session(profile_name=profile)
        s3_client = session.client("s3")
        
        # Test connection
        sts_client = session.client("sts")
        identity = sts_client.get_caller_identity()
        logger.info(f"AWS identity verified: {identity.get('Arn')}")
        
        return s3_client
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        print(f"Error: {e}")
        print("\n*** YOUR AWS SSO TOKEN MAY HAVE EXPIRED ***")
        print("Try running 'aws sso login' and then retry this script.")
        sys.exit(1)

def download_s3_directory(s3_client, bucket, prefix, local_dir):
    """Download an entire directory from S3."""
    # Make sure the local directory exists
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    files_downloaded = 0
    
    # List objects in the S3 bucket with the given prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            logger.warning(f"No objects found with prefix: {prefix}")
            return files_downloaded
        
        for obj in page["Contents"]:
            # Get the key (file path)
            key = obj["Key"]
            
            # Calculate relative path
            relative_path = key[len(prefix):]
            if relative_path.startswith("/"):
                relative_path = relative_path[1:]
            
            if not relative_path:  # Skip directories
                continue
            
            # Create the local file path
            local_file_path = local_path / relative_path
            
            # Make sure the directory exists
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download the file
            logger.info(f"Downloading {key} to {local_file_path}")
            s3_client.download_file(bucket, key, str(local_file_path))
            files_downloaded += 1
    
    logger.info(f"Downloaded {files_downloaded} files from {prefix} to {local_dir}")
    return files_downloaded

def download_quote_data(s3_client, quote_uuid, bucket=S3_BUCKET, data_dir=DATA_DIR):
    """Download all data for a quote from S3."""
    # Create local directory
    quote_dir = Path(data_dir) / quote_uuid
    quote_dir.mkdir(parents=True, exist_ok=True)
    
    # S3 prefix for this quote
    prefix = f"{quote_uuid}/"
    
    try:
        # Download the entire directory
        files_downloaded = download_s3_directory(s3_client, bucket, prefix, str(quote_dir))
        
        print(f"Successfully downloaded {files_downloaded} files for quote {quote_uuid}")
        print(f"Files saved to: {quote_dir}")
        return True
    except Exception as e:
        logger.error(f"Error downloading quote data: {e}")
        print(f"Error downloading quote data: {e}")
        return False

def list_quotes(s3_client, bucket=S3_BUCKET, limit=10):
    """List available quotes in the S3 bucket."""
    try:
        # List objects at the root level to find quote directories
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Delimiter='/',
            MaxKeys=limit
        )
        
        # Extract common prefixes (directories)
        if "CommonPrefixes" in response:
            # Each prefix is a directory like "12345/"
            quote_uuids = [
                prefix["Prefix"].rstrip('/') 
                for prefix in response["CommonPrefixes"]
            ]
            print(f"Found {len(quote_uuids)} quotes (showing up to {limit}):")
            for uuid in quote_uuids:
                print(f"  - {uuid}")
            return quote_uuids
        else:
            print("No quotes found in the bucket.")
            return []
    except Exception as e:
        logger.error(f"Error listing quotes: {e}")
        print(f"Error listing quotes: {e}")
        return []

if __name__ == "__main__":
    # Initialize S3 client using SSO profile
    s3_client = init_s3_client(AWS_PROFILE)
    
    # Download the quote specified by QUOTE_UUID
    print(f"Downloading quote {QUOTE_UUID}...")
    download_quote_data(s3_client, QUOTE_UUID, S3_BUCKET, DATA_DIR)