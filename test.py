import os
from google import genai
from dotenv import load_dotenv

load_dotenv()


def list_my_models():
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    print("--- רשימת המודלים הזמינים עבורך ---")
    try:
        # בגרסה החדשה, אנחנו פשוט מדפיסים את האובייקט או את השם
        for model in client.models.list():
            print(f"Model Name: {model.name}")
    except Exception as e:
        print(f"שגיאה בסריקה: {e}")


if __name__ == "__main__":
    list_my_models()