import os
import zipfile
import chromadb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. فك ضغط قاعدة البيانات تلقائياً على السيرفر
if not os.path.exists("./biology_db/chroma.sqlite3") and os.path.exists("./biology_db.zip"):
    print("📦 جاري فك ضغط قاعدة البيانات...")
    with zipfile.ZipFile("./biology_db.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    print("✅ تم فك الضغط بنجاح!")

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
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={key}"
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        payload = {"content": {"parts": [{"text": text}]}}
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code == 200:
            return resp.json()["embedding"]["values"]
        else:
            raise Exception(f"Embed Error: {resp.text}")

def generate_text_response(prompt, key, client):
    # استخدام النموذج الحديث المعتمد حصرياً
    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return res.text
    except Exception:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
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
    
    has_matching_docs = False
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
                    source_file = metas[idx].get("source", "") if metas else ""
                    if ("أحياء" in req.subject and any(x in source_file for x in ["أحياء", "احياء", "الهيكل", "النفيس", "الشامل"])) or \
                       ("عرب" in req.subject and any(x in source_file for x in ["عرب", "نحو", "أدب", "بلاغة", "الأيام", "الكيان", "الأضواء", "الامتحان"])):
                        has_matching_docs = True
                        retrieved_context += f"\n--- [من المرجع: {source_file}] ---\n{doc}\n"
                        if source_file not in sources_used:
                            sources_used.append(source_file)
        except Exception as e:
            print(f"⚠️ تنبيه استعلام ChromaDB: {e}")

    universal_system_instruction = f"""
أنت أستاذ وموجه أول تربوي خبير في المنهج المصري للثانوية العامة لمادة ({req.subject}) لمسار ({'علمي علوم' if req.track == 'science' else 'علمي رياضة'}).
مهمتك الشرح التعليمي الدقيق لدرس "{req.lesson}" في مادة "{req.subject}".

[حالة المراجع في قاعدة البيانات]:
{f"تم العثور على مراجع مخصصة لمادة {req.subject} وهي: " + str(sources_used) if has_matching_docs else f"⚠️ تنبيه: لا توجد مذكرات مرفوعة حالياً تخص مادة ({req.subject}) في قاعدة البيانات."}

[قواعد الشرح التعليمي الصارم]:
1. إذا كانت المراجع تخص مادة {req.subject}، اعتمد عليها.
2. إذا لم تكن هناك مراجع تخص {req.subject} في قاعدة البيانات، ضع في بداية الرد تنبيهاً مباشراً:
   '> **⚠️ تنبيه المراجع:** لم نجد مذكرات مرفوعة لمادة ({req.subject}) في قاعدة البيانات حتى الآن. تم إعداد هذا الشرح بالاعتماد المباشر على كتاب وزارة التربية والتعليم المعتمد.'
   ثم اشرح الدرس بالكامل بأعلى درجات الدقة والتفصيل وفقاً لكتاب الوزارة.
3. التزم بجميع القواعد الرسمية والتريكات وأفكار الامتحانات المعتمدة.
4. اختتم الشرح بـ "مفكرة التلخيص السريع".

[المحتوى المستخرج من المراجع]:
{retrieved_context if has_matching_docs else "لا توجد نصوص مرفوعة لهذه المادة."}
"""

    if req.user_query.strip():
        prompt = f"{universal_system_instruction}\n\nسؤال الطالب: '{req.user_query}'\nأجب بدقة مع التريكات."
    else:
        prompt = f"""{universal_system_instruction}

الهيكل الإجباري للشرح:
### شرح درس: {req.lesson} ({req.subject})
1. التمهيد والتأسيس التراكمي المطلوب.
2. الشرح والتفكيك التفصيلي للقواعد والمفاهيم.
3. ⚠️ تريكات وأفكار امتحانات المادة.
4. مفكرة التلخيص السريع.
"""

    try:
        explanation_text = generate_text_response(prompt, req.api_key, client)
        return {
            "status": "success",
            "explanation": explanation_text,
            "sources": sources_used if has_matching_docs else ["منهج وزارة التربية والتعليم الرسمي (لم يتم رفع مذكرات للمادة بعد)"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ أثناء التوليد: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
