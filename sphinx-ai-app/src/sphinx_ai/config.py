from sphinx_ai.utils.secrets import (get_aws_key_id, get_aws_region,
                                     get_aws_secret_access_key,
                                     get_db_location, get_input_dir,
                                     get_output_dir,
                                     get_output_transcript_bucket,
                                     get_twilio_account_sid,
                                     get_twilio_auth_token)

OUTPUT_DIR = get_output_dir()
INPUT_DIR = get_input_dir()
DB_LOCATION = get_db_location()
OUTPUT_TRANSCRIPTIONS_BUCKET_NAME = get_output_transcript_bucket()

TWILIO_ACCOUNT_SID=get_twilio_account_sid()
TWILIO_AUTH_TOKEN=get_twilio_auth_token()

AWS_ACCESS_KEY_ID=get_aws_key_id()
AWS_SECRET_ACCESS_KEY=get_aws_secret_access_key()
AWS_REGION=get_aws_region()

assert AWS_REGION is not None