import os.path
import io
import logging

# Google Drive API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Text extraction imports
import PyPDF2
import docx

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# --- Modified Text Extraction Functions ---
# These now accept file_content as bytes instead of a file_path

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file content."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            except Exception as e:
                logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF content: {str(e)}")
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from Word document content."""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        logger.error(f"Error extracting text from DOCX content: {str(e)}")
        return ""

def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from text file content."""
    try:
        return file_content.decode('utf-8').strip()
    except Exception as e:
        logger.error(f"Error extracting text from TXT content: {str(e)}")
        return ""

# --- Google Drive API Functions ---

def get_drive_service():
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

def list_drive_files(service, folder_id: str, page_size=25):
    """Lists files and folders from Google Drive."""
    try:
        query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"

        results = (
            service.files()
            .list(
                q=query,
                pageSize=page_size,
                fields="nextPageToken, files(id, name, mimeType)"
            ).execute()
        )
        items = results.get("files", [])
        if not items:
            print(f"No files found in folder with ID: {folder_id}.")
            return
        print("--- Files found in folder ---")
        for item in items:
            print(f"Name: {item['name']} (ID: {item['id']})")
        print("----------------------------------------")
    except HttpError as error:
        logger.error(f"An error occurred listing files: {error}")

def download_and_extract_text(service, file_id: str) -> str:
    """Downloads a file from Drive and extracts its text."""
    try:
        # Get file metadata to determine its type
        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata.get('name')
        mime_type = file_metadata.get('mimeType')
        
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
            return extract_text_from_pdf(file_content)
        elif file_content_type in ['.docx', '.doc']:
            return extract_text_from_docx(file_content)
        elif file_content_type == '.txt':
            return extract_text_from_txt(file_content)
        else:
            logger.warning(f"No text extractor for file type: {file_content_type}")
            return ""

    except HttpError as error:
        logger.error(f"An error occurred downloading/extracting file {file_id}: {error}")
        return ""

def main():
    """Main function to run the Drive test."""
    service = get_drive_service()
    if not service:
        return

    folder_id = input("Enter the Google Drive Folder ID of the document you want to read: ")
    if not folder_id:
        print("No Folder ID entered. Exiting.")
        return
    
    print("\n--- Listing Files ---")
    list_drive_files(service, folder_id=folder_id.strip())

    file_id = input("\nEnter the File ID of the document you want to read: ")
    if not file_id:
        print("No File ID entered. Exiting.")
        return
        
    print("\n--- Extracting Text ---")
    extracted_text = download_and_extract_text(service, file_id.strip())
    
    if extracted_text:
        print("\n--- Extracted Text ---")
        print(extracted_text)
    else:
        print("\nCould not extract text from the specified file.")

if __name__ == "__main__":
    main()