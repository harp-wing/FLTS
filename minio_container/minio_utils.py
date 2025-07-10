# import boto3 # type: ignore
# import os
# from botocore.client import Config # type: ignore
# from botocore.exceptions import ClientError # type: ignore
# from config import settings

# AWS_SECRET_KEY = settings.aws_secret_access_key
# S3_ENDPOINT = settings.aws_s3_endpoint
# BUCKET_NAME = settings.bucket_name
# AWS_ACCESS_KEY = settings.aws_access_key_id

# # Create S3 client
# s3 = boto3.client(
#     "s3",
#     endpoint_url=S3_ENDPOINT,
#     aws_access_key_id=AWS_ACCESS_KEY,
#     aws_secret_access_key=AWS_SECRET_KEY,
#     config=Config(signature_version="s3v4"),
#     region_name="us-west-1"
# )

# # Ensure bucket exists
# def ensure_bucket():
#     try:
#         s3.head_bucket(Bucket=BUCKET_NAME)
#     except ClientError:
#         s3.create_bucket(Bucket=BUCKET_NAME)

# def upload_file(file_obj, filename: str):
#     s3.upload_fileobj(file_obj, BUCKET_NAME, filename)

# def download_file(filename: str):
#     return s3.get_object(Bucket=BUCKET_NAME, Key=filename)["Body"]

import io
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from minio import Minio
from minio.error import S3Error
import os

# --- FastAPI App Initialization ---
app = FastAPI()

# --- MinIO Client Configuration ---
# It's recommended to use environment variables for sensitive data
# In a real-world scenario, you would set these in your container environment
load_dotenv()
ENDPOINT = os.getenv("ENDPOINT")
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
BUCKET = os.getenv("BUCKET")

# Initialize the MinIO client
# The 'secure' flag should be False if you are running MinIO locally without TLS
try:
    client = Minio(
        ENDPOINT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False
    )
except Exception as e:
    # This helps in debugging connection issues at startup
    print(f"Error initializing MinIO client: {e}")
    client = None

@app.get('/download/{object_name:path}')
async def get_file_from_minio(object_name: str = Path(..., description="The full path/name of the object to retrieve from the bucket.")):
    """
    API endpoint to retrieve any file from a MinIO bucket and stream it.
    The object name is passed as part of the URL path.
    Example: /download/my_data.csv
    Example: /download/images/photo.jpg
    """
    if not client:
        raise HTTPException(status_code=503, detail="MinIO client is not initialized.")

    try:
        # Get the object data from MinIO
        response = client.get_object(BUCKET, object_name)
        
        # Guess the MIME type of the file based on its extension
        media_type, _ = mimetypes.guess_type(object_name)
        if media_type is None:
            media_type = 'application/octet-stream' # Default for unknown file types

        # Create a streaming response, which is memory-efficient
        return StreamingResponse(
            response.stream(32*1024), # Stream in 32KB chunks
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=\"{os.path.basename(object_name)}\""}
        )

    except S3Error as exc:
        # Handle MinIO-specific errors
        if exc.code == 'NoSuchKey':
            raise HTTPException(status_code=404, detail=f"Object '{object_name}' not found in bucket '{BUCKET}'.")
        else:
            raise HTTPException(status_code=500, detail=f"MinIO S3 Error: {str(exc)}")
    except Exception as e:
        # Handle other potential errors
        # Make sure to release the connection on error
        if 'response' in locals() and response:
            response.close()
            response.release_conn()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# To run this app, you'll use an ASGI server like uvicorn:
# uvicorn app:app --host 0.0.0.0 --port 5000 --reload