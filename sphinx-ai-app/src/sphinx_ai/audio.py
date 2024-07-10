import asyncio
import json
import urllib
from datetime import datetime

import boto3  # type: ignore
from botocore.exceptions import ClientError

from sphinx_ai.config import (AWS_ACCESS_KEY_ID, AWS_REGION,
                              AWS_SECRET_ACCESS_KEY)
from sphinx_ai.utils.logging_config import logger

aws_logger = logger.bind(aws_key_id=AWS_ACCESS_KEY_ID, aws_secret_id=AWS_SECRET_ACCESS_KEY)

# def create_semi_random_names(initial_string):
#     # The generated bucket name must be between 3 and 63 chars long
#     return "".join([initial_string, str(uuid.uuid4())])

class S3Manager():
    def __init__(self, s3_bucket_name:str, ):
        self.session= boto3.session.Session( # type: ignore
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
        aws_logger.info(f'Logged in to AWS using AWS KEY ID: {AWS_ACCESS_KEY_ID}')
        self.current_region = AWS_REGION
        self.client = self.session.client("s3")
        self.resource = self.session.resource("s3")
        self.s3_connection=self.resource.meta.client
        
        self.s3_bucket_name = s3_bucket_name
        self.s3_bucket_uri = f's3://{self.s3_bucket_name}'
        self.s3_bucket_arn = 'arn:aws:s3:::sphinxai-audio-transcripts' #f"arn:aws:s3:::{self.s3_bucket_name}" 


    def create_bucket_if_not_exists(self):
        """Creates an S3 bucket in the specified region if it doesn't exist.

        Args:
            bucket_name: The name of the bucket to create.
            region: (Optional) The region to create the bucket in. If not specified, 
                    the default region will be used.

        Returns:
            The bucket's URI if the bucket exists or was successfully created, 
            or None if an error occurred.
        """
        
        location = {'LocationConstraint': self.current_region}

        try:
            self.client.head_bucket(Bucket=self.s3_bucket_uri)  # Check if bucket exists
            aws_logger.info(f"Bucket '{self.s3_bucket_uri}' already exists.")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                aws_logger.info(f"Creating bucket '{self.s3_bucket_uri}'...")
                self.client.create_bucket(Bucket=self.s3_bucket_uri, CreateBucketConfiguration=location)
            else:
                aws_logger.info(f"Error checking/creating bucket: {e}")
                return None

        aws_logger.info(f"Bucket URI: {self.s3_bucket_uri}")
        return self.s3_bucket_uri


    # upload file
    def upload_file(self, file_path):
        self.client.upload_file(
            Filename=file_path, # "/home/martin/Projects/ongoing/sphinx-ai/miscellaneous/azul_rojo.mp3",
            Bucket=self.s3_bucket_name,
            Key=file_path.split('/')[-1],
        )

        return f"{self.s3_bucket_uri}/{file_path.split('/')[-1]}"


class SpeechToText():
    def __init__(self, output_backet_uri:str, transc_lang:str='es-US'):
        self.s3_manager = S3Manager(output_backet_uri)
        self.output_backet_uri = output_backet_uri
        self.transc_lang = transc_lang
        self.__initialize_transcriber()


    def __initialize_transcriber(self):
        # self.s3_manager.create_bucket_if_not_exists(self.output_backet_name)
        self.transcribe = boto3.client(
            "transcribe",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=self.s3_manager.current_region,
        )
        aws_logger.info("Transcriber initialized")


    async def __transcribe_s3_file(self, file_s3_uri):
        try:
            self.transcribe = boto3.client('transcribe')
            job_name = f"audio_transcription_{datetime.today().strftime('%Y_%m_%d_%H_%M_%S')}"
            self.transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={"MediaFileUri": file_s3_uri},
                MediaFormat="mp3",
                LanguageCode=self.transc_lang,
            )

        except ClientError as e:
            aws_logger.info(f"Error starting transcription job: {e}")
            return

        while True:
            status = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                aws_logger.info(status['TranscriptionJob']['TranscriptionJobStatus'])
                break
            await asyncio.sleep(5)  # Poll every 5 seconds

        # os.remove("temp_audio.wav")
        
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            transcript = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
            response = urllib.request.urlopen(transcript['TranscriptionJob']['Transcript']['TranscriptFileUri'])
            data = json.loads(response.read())
            transcript_text = data["results"]["transcripts"][0]["transcript"]
            aws_logger.info(transcript_text)

            return transcript_text
        else:
            return "Transcription failed."


    # async def __transcribe_s3_file(self, file_s3_uri):
    #     self.__initialize_transcriber()
    #     self.last_job_name = f'audio_transcription_{datetime.now().strftime(('%Y-%m-%d %H:%M:%S'))}'
    #     aws_logger.info(self.last_job_name)
    #     self.transcribe.start_transcription_job(
    #         TranscriptionJobName=self.last_job_name,
    #         Media={"MediaFileUri": file_s3_uri},
    #         MediaFormat="mp3",
    #         LanguageCode="es-US",
    #     )
    #     while True:
    #         status = self.transcribe.get_transcription_job(TranscriptionJobName=self.last_job_name)
    #         if status["TranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
    #             break
    #         aws_logger.info("Not ready yet...", )
    #         time.sleep(2)

    #     if status["TranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
    #         response = urllib.request.urlopen(
    #             status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    #         )
    #         data = json.loads(response.read())
    #         text = data["results"]["transcripts"][0]["transcript"]
            
    #     return text
    
    async def transcribe_file(self, file_path):
        file_s3_uri = self.s3_manager.upload_file(file_path)
        aws_logger.info(file_s3_uri)
        transcription = await asyncio.gather(self.__transcribe_s3_file(file_s3_uri))

        if len(transcription) == 1:
            return transcription[0]