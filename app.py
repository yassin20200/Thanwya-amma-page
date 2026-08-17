import os
import zipfile
import chromadb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. فك ضغط قاعدة البيانات تلقائياً على السيرفر إذا كان الملف مضغوطاً
if not os.path.exists("./biology_db/chroma.sqlite3") and os.path.exists("./biology_db.zip"):
    print("📦 جاري فك ضغط قاعدة البيانات...")
    with zipfile.ZipFile("./biology_db.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    print("✅ تم فك الضغط وجاهز للاستخدام!")

app = FastAPI(title="Thanwya Amma Universal Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "./biology_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)

try:
    collection = chroma_client.get_collection(name="biology_materials")
except Exception:
    collection = None

class StudyRequest(BaseModel):
    subject: str
    lesson: str
    track: str = "science"
    user_query: str = ""
    api_key: str

def get_embed_vector(text, key, client):
    try:
        res = client.models.embed_content(
            model="models/gemini-embedding-2",
            contents=text
        )
        return res.embeddings[0].values
    except Exception:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        payload = {"content": {"parts": [{"text": text}]}}
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code == 200:
            return resp.json()["embedding"]["values"]
        else:
            raise Exception(f"Embed Error: {resp.text}")

def generate_text_response(prompt, key, client):
    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return res.text
    except Exception:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise Exception(f"Generate Error: {resp.text}")

@app.get("/")
def home():
    return {"status": "online", "message": "Thanwya Universal Engine is Ready!"}

@app.post("/api/explain")
async def explain_lesson(req: StudyRequest):
    if not req.api_key:
        raise HTTPException(status_code=400, detail="يرجى توفير API Key صحيح.")
    
    client = genai.Client(
        api_key=req.api_key,
        http_options=types.HttpOptions(headers={"x-goog-api-key": req.api_key})
    )
    
    search_text = f"مادة {req.subject} درس {req.lesson} {req.user_query}"
    retrieved_context = ""
    sources_used = []
    
    if collection and collection.count() > 0:
        try:
            query_vector = get_embed_vector(search_text, req.api_key, client)
            search_results = collection.query(
                query_embeddings=[query_vector],
                n_results=4
            )
            
            docs = search_results.get("documents", [[]])[0]
            metas = search_results.get("metadatas", [[]])[0]
            
            if docs:
                for idx, doc in enumerate(docs):
                    source_file = metas[idx].get("source", "مرجع عام") if metas else "مرجع"
                    retrieved_context += f"\n--- [من المرجع: {source_file}] ---\n{doc}\n"
                    if source_file not in sources_used:
                        sources_used.append(source_file)
        except Exception as e:
            print(f"⚠️ تنبيه ChromaDB: {e}")

    universal_system_instruction = f"""
أنت موجه أول وخبير تربوي في المنهج المصري للثانوية العامة لمادة ({req.subject}) لمسار ({'علمي علوم' if req.track == 'science' else 'علمي رياضة'}).
مهمتك الشرح الوافي والدقيق لدرس "{req.lesson}" في مادة "{req.subject}".

[قواعد الشرح والتنفيذ الصارم]:
1. اشرح درس "{req.lesson}" في مادة "{req.subject}" شرحاً تعليمياً شاملاً وتفصيلياً ومبسطاً وفقاً لمنهج وزارة التربية والتعليم المصرية للثانوية العامة.
2. إذا كانت المراجع المرفوعة أدناه تحتوي على معلومات مفيدة للمادة، استند إليها واعتمد عليها.
3. إذا كانت المراجع المرفوعة تخص مادة أخرى أو لا تغطي الدرس بالكامل، اشرح الدرس بالكامل من واقع معرفتك التامة والعميقة بالمنهج المصري الرسمي للثانوية العامة لمادة "{req.subject}". لا تعتذر ولا ترفض الشرح أبداً.
4. التزم بجميع المصطلحات الرسمية، والتريكات، والنقاط الحرجة وأسئلة الامتحانات المعتمدة للمادة.
5. اختتم الشرح بقسم "مفكرة التلخيص السريع".

[المراجع المتاحة في قاعدة البيانات]:
{retrieved_context if retrieved_context else "اعتمد على المنهج المصري المعتمد لكتاب الوزارة مباشرة."}
"""

    if req.user_query.strip():
        prompt = f"{universal_system_instruction}\n\nسؤال/استفسار الطالب المحدد: '{req.user_query}'\nأجب بدقة متناهية مع توضيح فكرة السؤال والتريكة المعتمدة فيه بالمنهج."
    else:
        prompt = f"""{universal_system_instruction}

قم بشرح الدرس طبقاً للهيكل التفاعلي التالي:
1. التمهيد والتأسيس التراكمي المطلوب للدرس.
2. الشرح التفصيلي للمفاهيم والقوانين وتفكيك القواعد بالتفصيل.
3. قسم خاص بعنوان "⚠️ تريكات وأفكار امتحانات المادة" يوضح النقاط التي تتكرر فيها الأخطاء بأسئلة الامتحانات.
4. قسم خاص بعنوان "مفكرة التلخيص السريع" يضم الملخص المباشر المجهز للحفظ والمراجعة.
"""

    try:
        explanation_text = generate_text_response(prompt, req.api_key, client)
        return {
            "status": "success",
            "explanation": explanation_text,
            "sources": sources_used
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ أثناء التوليد: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
