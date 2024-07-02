from pydantic import BaseModel

class DirectoryInput(BaseModel):
    directory : str

class QuestionInput(BaseModel):
    question: str