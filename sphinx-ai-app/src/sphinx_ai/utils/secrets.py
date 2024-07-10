import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="../.env", verbose=True)

def get_output_dir():
    return os.getenv("OUTPUT_DIR")

def get_input_dir():
    return os.getenv("INPUT_DIR")

def get_db_location():
    return os.getenv("DB_LOCATION")

def get_output_transcript_bucket():
    return os.getenv("OUTPUT_TRANSCRIPTIONS_BUCKET_NAME")

def get_aws_key_id():
    return os.getenv("AWS_ACCESS_KEY_ID")

def get_aws_secret_access_key():
    return os.getenv("AWS_SECRET_ACCESS_KEY")

def get_aws_region():
    return os.getenv("AWS_REGION")

def get_twilio_account_sid():
    return os.getenv("TWILIO_ACCOUNT_SID")

def get_twilio_auth_token():
    return os.getenv("TWILIO_AUTH_TOKEN")
