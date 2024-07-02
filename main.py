from fastapi import FastAPI, HTTPException,File,UploadFile
from pydantic import BaseModel
import pdfLLM
from models import DirectoryInput, QuestionInput
import os,shutil
from typing import List


# Intialize the FastAPI app
app = FastAPI()

# Directory to save the uploaded files
UPLOAD_DIR = 'uploads'


@app.post('/load_documents')
async def load_documents(files: List[UploadFile] = File(...)):
    try:
        # Clear the existing file in the firectory
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)

        # Recreate the directory
        os.makedirs(UPLOAD_DIR,exist_ok=True)
        saved_files = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIR,file.filename)
            with open(file_path,'wb') as buffer:
                buffer.write(await file.read())

            saved_files.append(file_path)

        return {
            'message': 'succesfully uploades'
        }

    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

@app.post('/start')
async def start_process():
    try:
        message = pdfLLM.load_documents(UPLOAD_DIR)
        return {
            'message': message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post('/ask_questions')
async def ask_question(questionInput: QuestionInput):
    try:
        answer = pdfLLM.ask_question(questionInput.question)
        return {
            'answer': answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))