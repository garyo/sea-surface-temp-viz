#!/usr/bin/env python3
"""Upload sea surface temperature map files to AWS S3."""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


def upload_file_to_s3(s3_client, local_path, bucket, s3_key, dry_run=False):
    """Upload a single file to S3."""
    if dry_run:
        print(f"[DRY RUN] Would upload: {local_path} -> s3://{bucket}/{s3_key}")
        return

    try:
        print(f"Uploading: {local_path} -> s3://{bucket}/{s3_key}")
        s3_client.upload_file(
            str(local_path),
            bucket,
            s3_key,
            # Note: Not setting ACL - bucket policy should control public access
        )
        print(f"  ✓ Uploaded successfully")
    except Exception as e:
        print(f"  ✗ Error uploading {local_path}: {e}")
        raise


def delete_file_from_s3(bucket, s3_prefix, filename, dry_run=False, aws_access_key=None, aws_secret_key=None):
    """Delete a file from S3 by filename."""
    # Create S3 client
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )
    else:
        s3_client = boto3.client("s3")

    # Construct the full S3 key
    s3_key = f"{s3_prefix}/{filename}" if s3_prefix else filename

    if dry_run:
        print(f"[DRY RUN] Would delete: s3://{bucket}/{s3_key}")
        return

    try:
        print(f"Deleting: s3://{bucket}/{s3_key}")
        s3_client.delete_object(Bucket=bucket, Key=s3_key)
        print(f"  ✓ Deleted successfully")

        # After deleting, regenerate and upload index.json
        print()
        print("Regenerating index.json...")
        index = generate_index_from_s3(bucket, s3_prefix, aws_access_key, aws_secret_key)

        # Save index locally to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(index, f, indent=2)
            temp_path = Path(f.name)

        # Upload updated index
        index_s3_key = f"{s3_prefix}/index.json" if s3_prefix else "index.json"
        upload_file_to_s3(s3_client, temp_path, bucket, index_s3_key, dry_run=False)

        # Clean up temp file
        temp_path.unlink()

        print()
        print("✓ File deleted and index.json updated")
    except Exception as e:
        print(f"  ✗ Error deleting file: {e}")
        raise


def list_bucket_contents(bucket, s3_prefix, aws_access_key=None, aws_secret_key=None):
    """List all files in the S3 bucket with the given prefix."""
    # Create S3 client
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )
    else:
        s3_client = boto3.client("s3")

    print(f"Listing contents of s3://{bucket}/{s3_prefix}/")
    print()

    try:
        # List objects with pagination support
        paginator = s3_client.get_paginator("list_objects_v2")
        prefix = f"{s3_prefix}/" if s3_prefix and not s3_prefix.endswith("/") else s3_prefix or ""

        total_size = 0
        file_count = 0
        files = []

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                files.append(obj)
                file_count += 1
                total_size += obj["Size"]

        if not files:
            print("No files found.")
            return

        # Sort by key name
        files.sort(key=lambda x: x["Key"])

        # Print files
        for obj in files:
            size_mb = obj["Size"] / (1024 * 1024)
            modified = obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S")
            print(f"{modified}  {size_mb:8.2f} MB  {obj['Key']}")

        print()
        print(f"Total: {file_count} files, {total_size / (1024 * 1024):.2f} MB")

    except Exception as e:
        print(f"Error listing bucket contents: {e}")
        sys.exit(1)


def validate_data_completeness(s3_client, bucket, s3_prefix):
    """Validate that data is complete and warn about missing dates or datasets."""
    # List objects with pagination support
    paginator = s3_client.get_paginator("list_objects_v2")
    prefix = f"{s3_prefix}/" if s3_prefix and not s3_prefix.endswith("/") else s3_prefix or ""

    # Track which dates have which datasets
    sst_dates = set()
    anom_dates = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            # Extract just the filename from the full S3 key
            filename = obj["Key"].split("/")[-1]

            # Look for sst-temp files
            match = re.match(r'(\d{4}-\d{2}-\d{2})-sst-temp-equirect\.webp$', filename)
            if match:
                sst_dates.add(match.group(1))

            # Look for anomaly files
            match = re.match(r'(\d{4}-\d{2}-\d{2})-sst-temp-anomaly-equirect\.webp$', filename)
            if match:
                anom_dates.add(match.group(1))

    if not sst_dates and not anom_dates:
        print("⚠️  Warning: No dated texture files found in S3")
        return

    # Check for mismatched datasets
    all_dates = sst_dates | anom_dates
    sst_only = sst_dates - anom_dates
    anom_only = anom_dates - sst_dates

    if sst_only:
        print(f"⚠️  Warning: {len(sst_only)} date(s) have SST data but missing anomaly data:")
        for date in sorted(sst_only)[:10]:  # Show first 10
            print(f"     - {date}")
        if len(sst_only) > 10:
            print(f"     ... and {len(sst_only) - 10} more")

    if anom_only:
        print(f"⚠️  Warning: {len(anom_only)} date(s) have anomaly data but missing SST data:")
        for date in sorted(anom_only)[:10]:  # Show first 10
            print(f"     - {date}")
        if len(anom_only) > 10:
            print(f"     ... and {len(anom_only) - 10} more")

    # Check for gaps in date sequence (only for dates with complete data)
    complete_dates = sorted(list(sst_dates & anom_dates))
    if len(complete_dates) < 2:
        return

    missing_dates = []
    start_date = datetime.strptime(complete_dates[0], "%Y-%m-%d")
    end_date = datetime.strptime(complete_dates[-1], "%Y-%m-%d")
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str not in complete_dates:
            missing_dates.append(date_str)
        current_date += timedelta(days=1)

    if missing_dates:
        print(f"⚠️  Warning: {len(missing_dates)} missing date(s) in sequence from {complete_dates[0]} to {complete_dates[-1]}:")
        for date in missing_dates[:10]:  # Show first 10
            print(f"     - {date}")
        if len(missing_dates) > 10:
            print(f"     ... and {len(missing_dates) - 10} more")


def generate_index_from_s3(bucket, s3_prefix, aws_access_key=None, aws_secret_key=None):
    """Generate index.json from S3 bucket contents."""
    # Create S3 client
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )
    else:
        s3_client = boto3.client("s3")

    # List objects with pagination support
    paginator = s3_client.get_paginator("list_objects_v2")
    prefix = f"{s3_prefix}/" if s3_prefix and not s3_prefix.endswith("/") else s3_prefix or ""

    # Find all dated texture files in S3
    dates = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            # Extract just the filename from the full S3 key
            filename = obj["Key"].split("/")[-1]
            # Look for pattern: YYYY-MM-DD-sst-temp-equirect.webp
            match = re.match(r'(\d{4}-\d{2}-\d{2})-sst-temp-equirect\.webp$', filename)
            if match:
                dates.add(match.group(1))

    # Create index with sorted dates (newest last)
    dates_list = sorted(list(dates))
    index = {
        'dates': dates_list,
        'latest': dates_list[-1] if dates_list else None
    }

    print(f"Generated index from S3 with {len(dates_list)} dates")
    if dates_list:
        print(f"  Date range: {dates_list[0]} to {dates_list[-1]}")
        print(f"  Latest: {index['latest']}")

    # Validate data completeness
    print()
    print("Validating data completeness...")
    validate_data_completeness(s3_client, bucket, s3_prefix)

    return index


def fetch_existing_objects_metadata(s3_client, bucket, s3_prefix):
    """Return metadata for existing objects under the prefix keyed by full S3 key."""
    paginator = s3_client.get_paginator("list_objects_v2")
    prefix = f"{s3_prefix}/" if s3_prefix and not s3_prefix.endswith("/") else s3_prefix or ""
    metadata = {}

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            metadata[obj["Key"]] = {
                "size": obj["Size"],
                "last_modified": obj["LastModified"],
            }

    return metadata


def should_skip_upload(
    s3_client,
    bucket,
    s3_key,
    local_path,
    debug=False,
    remote_metadata=None,
):
    """Return True when remote object matches size and is not older than local."""
    if remote_metadata is None:
        try:
            response = s3_client.head_object(Bucket=bucket, Key=s3_key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchKey"):
                return False
            raise
        remote_size = response.get("ContentLength")
        remote_last_modified = response.get("LastModified")
    else:
        remote_size = remote_metadata.get("size")
        remote_last_modified = remote_metadata.get("last_modified")

    if remote_size is None or remote_last_modified is None:
        return False

    local_stat = local_path.stat()
    local_size = local_stat.st_size
    local_mtime = datetime.fromtimestamp(local_stat.st_mtime, tz=timezone.utc)

    if debug:
        remote_ts = remote_last_modified.isoformat()
        local_ts = local_mtime.isoformat()
        print(
            f"[DEBUG] {local_path.name}: remote size={remote_size}, mtime={remote_ts}; "
            f"local size={local_size}, mtime={local_ts}"
        )

    if remote_size == local_size and remote_last_modified >= local_mtime:
        print(
            f"Skipping: {local_path} -> s3://{bucket}/{s3_key} "
            "(remote file exists, same size, and not older)"
        )
        return True

    if debug:
        if remote_size != local_size:
            print(f"[DEBUG] {local_path.name}: size differs (remote {remote_size} vs local {local_size})")
        elif remote_last_modified < local_mtime:
            print(f"[DEBUG] {local_path.name}: remote is older (remote {remote_last_modified.isoformat()} < local {local_mtime.isoformat()})")

    return False


def upload_maps_directory(
    maps_dir,
    bucket,
    s3_prefix,
    dry_run=False,
    aws_access_key=None,
    aws_secret_key=None,
    always_upload_maps=False,
    debug_skip_checks=False,
):
    """Upload all files in the maps directory to S3, then generate and upload index.json."""
    maps_path = Path(maps_dir)

    if not maps_path.exists():
        print(f"Error: Directory {maps_dir} does not exist")
        sys.exit(1)

    # Get all files to upload (excluding any existing index.json)
    files = [f for f in maps_path.glob("*") if f.is_file() and f.name != "index.json"]
    if not files:
        print(f"No files found in {maps_dir}")
        return

    # Create S3 client
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )
    else:
        # Use default credentials from environment or AWS config
        s3_client = boto3.client("s3")

    print(f"Found {len(files)} files to upload")
    print(f"Target: s3://{bucket}/{s3_prefix}")
    print()

    existing_objects = fetch_existing_objects_metadata(s3_client, bucket, s3_prefix)
    if debug_skip_checks:
        print(f"[DEBUG] Found {len(existing_objects)} existing object(s) in S3 under prefix")

    upload_count = 0
    skipped_count = 0

    # Upload each file
    for file_path in sorted(files):
        s3_key = f"{s3_prefix}/{file_path.name}" if s3_prefix else file_path.name
        if not always_upload_maps and should_skip_upload(
            s3_client,
            bucket,
            s3_key,
            file_path,
            debug=debug_skip_checks,
            remote_metadata=existing_objects.get(s3_key),
        ):
            skipped_count += 1
            continue
        upload_file_to_s3(s3_client, file_path, bucket, s3_key, dry_run)
        upload_count += 1

    print()
    print(f"{'[DRY RUN] Would upload' if dry_run else 'Uploaded'} {upload_count} file(s)")
    if skipped_count:
        print(f"Skipped {skipped_count} file(s) already present in S3")
    print()

    # Now generate index.json from S3 contents (includes what we just uploaded)
    print("Generating index.json from S3 bucket contents...")
    index = generate_index_from_s3(bucket, s3_prefix, aws_access_key, aws_secret_key)

    # Save index locally
    index_path = maps_path / 'index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"Saved index.json to {index_path}")

    # Upload index.json
    print()
    s3_key = f"{s3_prefix}/index.json" if s3_prefix else "index.json"
    upload_file_to_s3(s3_client, index_path, bucket, s3_key, dry_run)

    print()
    print(f"{'[DRY RUN] Complete!' if dry_run else 'Upload complete!'}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Upload sea surface temperature maps to AWS S3"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List contents of the S3 bucket instead of uploading",
    )
    parser.add_argument(
        "--delete",
        type=str,
        help="Delete a file from S3 by filename (e.g., '2024-01-15-sst-temp-equirect.webp')",
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default="./maps",
        help="Directory containing map files to upload (default: ./maps)",
    )
    parser.add_argument(
        "--bucket",
        default="climate-change-assets",
        help="S3 bucket name (default: climate-change-assets)",
    )
    parser.add_argument(
        "--prefix",
        default="sea-surface-temp",
        help="S3 key prefix (default: sea-surface-temp)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=".env",
        help="Path to .env file with AWS credentials (default: .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--always-upload-maps",
        action="store_true",
        help="Upload all map files regardless of existing files in S3",
    )
    parser.add_argument(
        "--debug-upload-checks",
        action="store_true",
        help="Print debug details about skip decisions",
    )

    args = parser.parse_args(argv)

    # Load environment variables from .env file
    if args.env_file.exists():
        print(f"Loading credentials from {args.env_file}")
        load_dotenv(args.env_file)
    else:
        print(f"Warning: {args.env_file} not found, will use default AWS credentials")

    # Get AWS credentials from environment
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key or not aws_secret_key:
        print("Warning: AWS credentials not found in .env file or environment")
        print("Will attempt to use default AWS credentials (from ~/.aws/credentials)")

    if args.list:
        list_bucket_contents(
            args.bucket,
            args.prefix,
            aws_access_key,
            aws_secret_key,
        )
    elif args.delete:
        delete_file_from_s3(
            args.bucket,
            args.prefix,
            args.delete,
            args.dry_run,
            aws_access_key,
            aws_secret_key,
        )
    else:
        upload_maps_directory(
            args.maps_dir,
            args.bucket,
            args.prefix,
            args.dry_run,
            aws_access_key,
            aws_secret_key,
            args.always_upload_maps,
            args.debug_upload_checks,
        )


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
