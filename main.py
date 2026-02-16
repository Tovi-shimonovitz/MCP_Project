import os
import ssl
import certifi
from google import genai  # הספרייה החדשה
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# --- 1. תיקון שגיאות SSL ותעודות ---
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# --- 2. טעינת משתני סביבה ---
load_dotenv()

# --- 3. הגדרת שרת FastAPI ---
app = FastAPI(title="AI Business Agent Service")

# --- 4. הגדרת CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 5. הגדרת מבנה הבקשה ---
class ChatRequest(BaseModel):
    message: str
    provider: str = "gemini"  # הוספנו ברירת מחדל כדי שלא תקבלי שגיאת Validation


# --- 6. פונקציית הסוכן המעודכנת ---
def call_gemini_agent(prompt: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    # יצירת הלקוח
    client = genai.Client(api_key=api_key)

    # כאן הגדרנו את כל ה"שכל" של הסוכן שלך
    agent_logic = """
    אתה סוכן AI מומחה לכתיבה עסקית ורשמית.
    תפקידך: לקבל טקסט גולמי (לעתים מבולגן או בסלנג) ולהפוך אותו למייל רשמי, מנומס ומקצועי בעברית.

    חוקים לעבודה:
    1. שמור על המסר המקורי של המשתמש, אל תמציא עובדות חדשות.
    2. השתמש בשפה גבוהה אך מודרנית.
    3. מבנה התשובה:
       - נושא המייל: [הצעה לנושא]
       - גוף המייל: [הטקסט המשופר]
    4. אם הטקסט של המשתמש לא ברור, שאל שאלת הבהרה קצרה.
    5. אל תשתמש בפנייה ישירה לנמען ולא בגוף נוכח, שמור על סגנון מאופק.
    6. השמט מילים פוגעניות או אלימות.
    7. כל פעם שאתה שם את התווים \n תחליף אותם בירידת שורה
    """

    # שליחת הבקשה בפורמט של הספרייה החדשה
    try:
        # שימוש במודל המדויק שמצאנו ברשימה שלך
        response = client.models.generate_content(
            model='gemini-2.5-flash',  # או 'gemini-3-flash-preview' אם את רוצה את הכי חדש
            contents=prompt,
            config={
                'system_instruction': agent_logic,
                'response_mime_type': 'application/json'  # מכריח את ה-AI לענות ב-JSON
            }
        )

        return response.text
    except Exception as e:
        print(f"Error during generation: {e}")
        raise e


# --- 7. ה-Endpoint ---
@app.post("/v1/chat")
async def chat_gateway(request: ChatRequest):
    try:
        if request.provider.lower() == "gemini":
            answer = call_gemini_agent(request.message)
            return {
                "status": "success",
                "provider": "gemini",
                "answer": answer
            }
        else:
            raise HTTPException(status_code=400, detail="Provider not supported")
    except Exception as e:
        print(f"Error: {str(e)}")  # הדפסה לטרמינל כדי שתוכלי לראות מה קרה
        raise HTTPException(status_code=500, detail=str(e))


# --- 8. בדיקת דופק ---
@app.get("/")
def health_check():
    return {"message": "Gateway is up and running!"}