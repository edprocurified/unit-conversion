import json
import logging
import os
from pathlib import Path

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

class S3Loader:
    """Class for loading quote data from S3 bucket."""
    
    def __init__(self, bucket=None, profile=None, data_dir="data"):
        """
        Initialize the S3 loader.
        
        Args:
            bucket (str): S3 bucket name (defaults to S3_BUCKET)
            profile (str): AWS profile name (defaults to AWS_PROFILE)
            data_dir (str): Local directory to store downloaded data
        """
        self.bucket = bucket or S3_BUCKET
        self.profile = profile or AWS_PROFILE
        self.data_dir = Path(data_dir)
        self.s3_client = self._init_s3_client()
        
    def _init_s3_client(self):
        """Initialize and return an S3 client using boto3."""
        try:
            # Try to create a boto3 session with the SSO profile
            session = boto3.Session(profile_name=self.profile)
            s3_client = session.client("s3")
            
            # Test connection by checking bucket access
            try:
                s3_client.head_bucket(Bucket=self.bucket)
                logger.info(f"Successfully connected to bucket: {self.bucket}")
                return s3_client
            except ClientError as e:
                # If bucket doesn't exist or no access
                logger.warning(f"Error accessing bucket '{self.bucket}': {e}")
                # Try to fallback to environment variables
                logger.info("Trying to use environment credentials...")
                default_session = boto3.Session()
                return default_session.client("s3")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    def download_s3_directory(self, prefix, local_dir):
        """
        Download an entire directory from S3.
        
        Args:
            prefix (str): S3 key prefix (directory path)
            local_dir (str): Local directory to save downloaded files
            
        Returns:
            int: Number of files downloaded
        """
        # Make sure the local directory exists
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        files_downloaded = 0
        
        # List objects in the S3 bucket with the given prefix
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
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
                self.s3_client.download_file(self.bucket, key, str(local_file_path))
                files_downloaded += 1
        
        logger.info(f"Downloaded {files_downloaded} files from {prefix} to {local_dir}")
        return files_downloaded
    
    def download_quote_data(self, quote_uuid):
        """
        Download all data for a quote from S3.
        
        Args:
            quote_uuid (str): UUID for the quote
            
        Returns:
            dict: Information about the downloaded data
        """
        # Create local directory
        quote_dir = self.data_dir / quote_uuid
        quote_dir.mkdir(parents=True, exist_ok=True)
        
        # S3 prefix for this quote
        prefix = f"{quote_uuid}/"
        
        try:
            # Download the entire directory
            files_downloaded = self.download_s3_directory(prefix, str(quote_dir))
            
            return {
                "status": "success",
                "message": f"Downloaded {files_downloaded} files for quote {quote_uuid}",
                "quote_dir": str(quote_dir),
                "files_downloaded": files_downloaded
            }
        except Exception as e:
            logger.error(f"Error downloading quote data: {e}")
            return {
                "status": "error", 
                "message": f"Error downloading quote data: {e}"
            }
    
    def extract_quote_info(self, quote_uuid):
        """
        Extract structured information about a quote, including line items.
        
        Args:
            quote_uuid (str): UUID for the quote
            
        Returns:
            dict: Structured information about the quote
        """
        # First ensure the quote data is downloaded
        quote_dir = self.data_dir / quote_uuid
        if not quote_dir.exists():
            download_result = self.download_quote_data(quote_uuid)
            if download_result["status"] != "success":
                return download_result
        
        # Look for results in the results directory
        results_dir = quote_dir / "results"
        if not results_dir.exists():
            return {
                "status": "error",
                "message": f"No results directory found for quote {quote_uuid}",
            }
        
        # Collect all result files
        result_files = list(results_dir.glob("result_*.json"))
        if not result_files:
            return {
                "status": "error",
                "message": f"No result files found for quote {quote_uuid}",
            }
        
        # Load and combine results
        line_items = []
        for result_file in result_files:
            try:
                with open(result_file, "r") as f:
                    result_data = json.load(f)
                    if "line_items" in result_data:
                        line_items.extend(result_data["line_items"])
            except Exception as e:
                logger.warning(f"Error loading result file {result_file}: {e}")
        
        return {
            "status": "success",
            "quote_uuid": quote_uuid,
            "quote_dir": str(quote_dir),
            "line_items": line_items,
            "item_count": len(line_items)
        }
        
    def list_available_quotes(self):
        """
        List available quotes in the S3 bucket.
        
        Returns:
            list: List of quote UUIDs
        """
        try:
            # List objects at the root level to find quote directories
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Delimiter='/'
            )
            
            # Extract common prefixes (directories)
            if "CommonPrefixes" in response:
                # Each prefix is a directory like "12345/"
                quote_uuids = [
                    prefix["Prefix"].rstrip('/') 
                    for prefix in response["CommonPrefixes"]
                ]
                return {
                    "status": "success",
                    "quote_uuids": quote_uuids,
                    "count": len(quote_uuids)
                }
            else:
                return {
                    "status": "success",
                    "quote_uuids": [],
                    "count": 0
                }
        except Exception as e:
            logger.error(f"Error listing quotes: {e}")
            return {
                "status": "error",
                "message": f"Error listing quotes: {e}"
            }


# Helper function to initialize the loader
def get_loader(bucket=None, profile=None, data_dir="data"):
    """Create and return an S3Loader instance."""
    return S3Loader(bucket=bucket, profile=profile, data_dir=data_dir)


if __name__ == "__main__":
    # Example usage
    loader = get_loader()
    
    # List available quotes
    quotes = loader.list_available_quotes()
    print(f"Found {quotes['count']} quotes in bucket {loader.bucket}")
    
    # Download a specific quote (if UUID provided)
    import sys
    if len(sys.argv) > 1:
        quote_uuid = sys.argv[1]
        print(f"Downloading quote {quote_uuid}...")
        result = loader.download_quote_data(quote_uuid)
        print(result["message"])
        
        # Extract info about the quote
        info = loader.extract_quote_info(quote_uuid)
        if info["status"] == "success":
            print(f"Found {info['item_count']} line items in quote {quote_uuid}")