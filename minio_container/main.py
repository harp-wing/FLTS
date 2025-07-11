# main.py

import os
import io
import mimetypes
from fastapi import FastAPI, UploadFile, File, HTTPException, Path, Depends, Request # type: ignore
from fastapi.responses import StreamingResponse, JSONResponse # type: ignore
from minio import Minio # type: ignore
from minio.error import S3Error # type: ignore
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MinIO Configuration ---
# Standardized environment variable names as requested.
ENDPOINT = os.getenv("ENDPOINT", "minio:9000")
ACCESS_KEY = os.getenv("ACCESS_KEY", "minioadmin")
SECRET_KEY = os.getenv("SECRET_KEY", "minioadmin")
BUCKET = os.getenv("BUCKET", "dataset") # Default bucket for startup check

app = FastAPI(
    title="MinIO File Gateway",
    description="An API to upload and download files from a MinIO bucket."
)

# --- MinIO Client Holder ---
# A simple way to manage the client instance.
minio_client = None

# --- Dependency Injection ---
def get_minio_client():
    """
    Provides a MinIO client instance.
    This function will be used by FastAPI's dependency injection system.
    """
    if not minio_client:
        raise HTTPException(status_code=503, detail="MinIO service is not available.")
    return minio_client

# --- Application Events ---
@app.on_event("startup")
def startup_event():
    """On startup, initialize the MinIO client and check for the default bucket."""
    global minio_client
    try:
        minio_client = Minio(
            ENDPOINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=False  # Set to True if using HTTPS
        )
        logger.info(f"Successfully connected to MinIO at {ENDPOINT}.")

        # Check if the default bucket exists and create it if it doesn't.
        found = minio_client.bucket_exists(BUCKET)
        if not found:
            minio_client.make_bucket(BUCKET)
            logger.info(f"Bucket '{BUCKET}' created.")
        else:
            logger.info(f"Bucket '{BUCKET}' already exists.")

        # ------------------------TO BE-------------------------
        # -----------------------REMOVED-------------------------

        # --- Upload local 'dataset' directory ---
        local_dataset_path = "dataset"
        if not os.path.isdir(local_dataset_path):
            logger.warning(f"Local directory '{local_dataset_path}' not found. Skipping initial upload.")
            return
            
        logger.info(f"Starting upload of directory '{local_dataset_path}' to bucket '{BUCKET}'...")

        for root, _, files in os.walk(local_dataset_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                
                # Create a relative path for the object name in MinIO to preserve directory structure.
                # Since you're on Windows, os.path.relpath will handle the path separators correctly.
                object_name = os.path.relpath(local_file_path, local_dataset_path)
                
                # Ensure forward slashes for the object name, which is standard for object storage.
                object_name = object_name.replace("\\", "/")

                try:
                    # Use fput_object for efficient file upload from a local path
                    minio_client.fput_object(
                        bucket_name=BUCKET,
                        object_name=object_name,
                        file_path=local_file_path,
                    )
                    logger.info(f"Successfully uploaded '{local_file_path}' as '{object_name}'")
                except S3Error as exc:
                    logger.error(f"Error uploading '{local_file_path}': {exc}")

    except Exception as e:
        logger.error(f"Error connecting to MinIO or creating bucket: {e}")
        # We set client to None to indicate failure. The dependency will then raise an error.
        minio_client = None
    


@app.post(
    "/upload/{bucket_name}/{object_name:path}",
    summary="Upload a file to MinIO",
    description="Receives binary data in the request body and uploads it to the specified MinIO bucket and object path.",
    tags=["Files"],
    responses={
        200: {"description": "File uploaded successfully."},
        400: {"description": "No data received in the request body."},
        503: {"description": "MinIO service is unavailable."},
    }
)
async def minio_upload(
    bucket_name: str,
    object_name: str,
    request: Request,
    client: Minio = Depends(get_minio_client)
):
    """
    Receives data from the request body and uploads it to a MinIO bucket.
    If the bucket does not exist, it will be created automatically.
    """
    try:
        # 1. Read the entire request body as bytes.
        data = await request.body()
        if not data:
            raise HTTPException(
                status_code=400, detail="No data received in request body."
            )
        
        data_len = len(data)
        # 2. Wrap the bytes data in a file-like object (io.BytesIO).
        data_stream = io.BytesIO(data)

        # 3. Ensure the target bucket exists.
        found = client.bucket_exists(bucket_name)
        if not found:
            logger.info(f"Bucket '{bucket_name}' not found. Creating it.")
            client.make_bucket(bucket_name)
        else:
            logger.info(f"Bucket '{bucket_name}' already exists.")

        # 4. Upload the object to MinIO.
        logger.info(f"Uploading '{object_name}' ({data_len} bytes) to bucket '{bucket_name}'...")
        client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=data_stream,
            length=data_len,
            content_type='application/octet-stream'
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "bucket": bucket_name,
                "object_name": object_name,
                "size_bytes": data_len
            }
        )

    except S3Error as exc:
        logger.error(f"An S3 error occurred during upload: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during upload: {exc.code}"
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )

@app.get(
    "/download/{bucket_name}/{object_name:path}",
    summary="Download a file from MinIO",
    description="Streams a file directly from the specified MinIO bucket and object path.",
    tags=["Files"],
    responses={
        200: {"description": "Successful file stream."},
        404: {"description": "File or bucket not found."},
        503: {"description": "MinIO service is unavailable."}
    }
)
async def minio_download(
    bucket_name: str,
    object_name: str,
    client: Minio = Depends(get_minio_client)
):
    """
    Retrieves an object from a MinIO bucket and returns it as a streaming response.
    """
    file_stream = None
    try:
        # 1. Get object statistics to check for existence and get metadata.
        client.stat_object(bucket_name, object_name)

        # 2. Get the object data stream from MinIO.
        file_stream = client.get_object(bucket_name, object_name)

        # 3. Guess the media type from the filename, or default.
        media_type, _ = mimetypes.guess_type(object_name)
        if media_type is None:
            media_type = "application/octet-stream"

        # 4. Create a StreamingResponse.
        return StreamingResponse(
            file_stream.stream(32 * 1024),  # Stream in 32KB chunks
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=\"{os.path.basename(object_name)}\""
            }
        )

    except S3Error as exc:
        if exc.code == "NoSuchKey" or exc.code == "NoSuchBucket":
            raise HTTPException(
                status_code=404,
                detail=f"Object '{object_name}' not found in bucket '{bucket_name}'."
            )
        else:
            logger.error(f"An S3 error occurred: {exc}")
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while fetching the file: {exc.code}"
            )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )



