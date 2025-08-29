import os
import io
import uuid
import logging
from datetime import datetime
from typing import List

# Google Drive API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Core libraries
import PyPDF2
import docx
from chunking_evaluation.chunking import RecursiveTokenChunker

# Local imports
from data_models import DocumentChunk

logger = logging.getLogger(__name__)
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

class DocumentHandler:
    """Handles document retrieval from drive and text extraction."""

    def __init__(self, folder_id: str):
        self.folder_id = folder_id
        self.service = self.get_drive_service()

    def get_drive_service(self):
        """Authenticates with Google Drive API and returns the service object."""
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        
        try:
            return build("drive", "v3", credentials=creds)
        except HttpError as error:
            logger.error(f"An error occurred building the Drive service: {error}")
            return None
        
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                    continue
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from Word document."""
        try:
            doc = docx.Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX content: {str(e)}")
            return ""

    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extract text from text file."""
        try:
            return file_content.decode('utf-8', errors='replace').strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT content: {str(e)}")
            return ""

    def extract_text(self, file_content: bytes, file_type: str) -> str:
        """Extracts text from various document types."""
        if file_type == 'application/pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return self.extract_text_from_docx(file_content)
        elif file_type == 'text/plain':
            return self.extract_text_from_txt(file_content)
        return ""
    
    def list_drive_files(self, service, folder_id: str, page_size=25):
        """Lists files and folders from Google Drive."""
        try:
            query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"

            results = (
                service.files()
                .list(
                    q=query,
                    pageSize=page_size,
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)"
                ).execute()
            )
            items = results.get("files", [])
            if not items:
                logger.info(f"No files found in the provided folder.")
                return
            logger.info("--- Files found in folder ---")
            for item in items:
                logger.info(f"Name: {item['name']} (ID: {item['id']})")
            logger.info("----------------------------------------")
            return items
        except HttpError as error:
            logger.error(f"An error occurred listing files: {error}")

    
    def download_and_extract_text(self, service, file_id: str) -> str:
        """Downloads a file from Drive and extracts its text."""
        try:
            # Get file metadata to determine its type
            file_metadata = service.files().get(fileId=file_id).execute()
            file_name = file_metadata.get('name')
            mime_type = file_metadata.get('mimeType')
            modified_time = file_metadata.get('modifiedTime')
            
            logger.info(f"Downloading '{file_name}' (MIME type: {mime_type})...")

            # Handle Google Docs, which need to be exported
            if mime_type.startswith('application/vnd.google-apps'):
                if mime_type == 'application/vnd.google-apps.document':
                    request = service.files().export_media(fileId=file_id, mimeType='text/plain')
                    file_content_type = '.txt'
                else:
                    logger.error(f"Unsupported Google App type: {mime_type}")
                    return ""
            # Handle standard files (PDF, DOCX, TXT)
            else:
                request = service.files().get_media(fileId=file_id)
                file_content_type = os.path.splitext(file_name)[1].lower()

            # Download the file content into memory
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                logger.info(f"Download {int(status.progress() * 100)}%.")
            
            file_content = fh.getvalue()
            
            # Extract text based on file type
            if file_content_type == '.pdf':
                return self.extract_text_from_pdf(file_content)
            elif file_content_type in ['.docx', '.doc']:
                return self.extract_text_from_docx(file_content)
            elif file_content_type == '.txt':
                return self.extract_text_from_txt(file_content)
            else:
                logger.warning(f"No text extractor for file type: {file_content_type}")
                return ""

        except HttpError as error:
            logger.error(f"An error occurred downloading/extracting file {file_id}: {error}")
            return ""

class MultiDocumentProcessor:
    """Handles processing of multiple document types."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc'}

    def __init__(self, folder_id: str):
        self.processed_documents = {}
        self.document_handler = DocumentHandler(folder_id=folder_id)

    def process_extracted_text(self, document_handler: DocumentHandler) -> List[DocumentChunk]:
        """Process all supported documents in a directory."""
        all_chunks = []
        service = document_handler.service
        retrieved_files = self.document_handler.list_drive_files(folder_id=document_handler.folder_id, service=service)

        if not retrieved_files:
            logger.error(f"No files found in the provided folder.")
            return all_chunks
        
        # Find all supported files recursively
        supported_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            supported_files.extend([f for f in retrieved_files if f["name"].lower().endswith(ext)])

        logger.info(f"Found {len(supported_files)} supported documents")
        
        for file in supported_files:
            try:
                text = self.document_handler.download_and_extract_text(service, file_id=file['id'])
                if text.strip():
                    # Pass the dict
                    chunks = self.chunk_document(file, text)
                    all_chunks.extend(chunks)
                    self.processed_documents[file['name']] = len(chunks)
                    logger.info(f"Processed {file['name']}: {len(chunks)} chunks")
                else:
                    logger.warning(f"No text extracted from {file['name']}")
            except Exception as e:
                logger.error(f"Error processing {file['name']}: {str(e)}")
                continue
        
        return all_chunks
    
    def chunk_document(self, file, text: str) -> List[DocumentChunk]:
        """Split document into chunks with source tracking."""
        text_splitter = RecursiveTokenChunker(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!"],
        )
        
        chunks = text_splitter.split_text(text)
        
        # Get last modified timestamp of the source document
        last_modified = file['modifiedTime']
        source_document = file['name']
        document_type = file['mimeType']

        if isinstance(last_modified, (int, float)):
            modified_time = datetime.fromtimestamp(last_modified).isoformat()
        else:
            modified_time = str(last_modified)

        document_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_obj = DocumentChunk(
                content=chunk,
                chunk_id=chunk_id,
                chunk_index=i,
                source_document=source_document,
                document_type=document_type,
                last_modified=modified_time
            )
            document_chunks.append(chunk_obj)
        
        return document_chunks
