import os
import zipfile
import chromadb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. فك ضغط قاعدة البيانات تلقائياً على السيرفر (أو تحميلها إذا كان هناك رابط DB_ZIP_URL)
DB_ZIP_URL = os.environ.get("https://github.com/yassin20200/Thanwya-amma-page/releases/download/v1.0/biology_db.2.zip")

if not os.path.exists("./biology_db/chroma.sqlite3"):
    if DB_ZIP_URL:
        try:
            print("📦 جاري تحميل قاعدة البيانات من الرابط الخارجي...")
            r = requests.get(DB_ZIP_URL, timeout=60)
            with open("biology_db.zip", "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile("biology_db.zip", "r") as zip_ref:
                zip_ref.extractall(".")
            print("✅ تم التحميل وفك الضغط بنجاح!")
        except Exception as e:
            print(f"⚠️ خطأ أثناء تحميل الـ ZIP: {e}")
    elif os.path.exists("./biology_db.zip"):
        print("📦 جاري فك ضغط biology_db.zip المحلي...")
        with zipfile.ZipFile("./biology_db.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        print("✅ تم فك الضغط بنجاح!")

app = FastAPI(title="Thanwya Amma Smart & Flexible RAG Engine")

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
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json()["embedding"]["values"]
        else:
            raise Exception(f"Embed Error: {resp.text}")

def generate_text_response(prompt, key, client):
    # تجربة النماذج بالترتيب لضمان الاستقرار الفائق
    models_to_try = ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-2.0-flash"]
    for model_name in models_to_try:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return res.text
        except Exception:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
                headers = {"Content-Type": "application/json", "x-goog-api-key": key}
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                continue
    raise Exception("تعذر توليد الإجابة من جميع النماذج المتاحة. يرجى مراجعة صلاحية المفتاح.")

def is_source_relevant(subject, source_file):
    """التحقق الذكي من تطابق المرجع مع المادة المطلوبة"""
    sub = subject.lower()
    src = source_file.lower()
    
    keywords = {
        "أحياء": ["أحياء", "احياء", "الهيكل", "النفيس", "الشامل", "مذكرة أحياء", "bio", "وراثة", "تضاعف"],
        "عرب": ["عرب", "نحو", "أدب", "بلاغة", "الأيام", "الكيان", "الأضواء", "الامتحان", "نصوص", "قراءة", "تعبير"],
        "فيزياء": ["فيزياء", "فيزيا", "physics", "كهربية", "كيرشوف", "أوم", "حديثة", "النيوتن", "دينامو", "حث"],
        "كيمياء": ["كيمياء", "كيميا", "chemistry", "عضوية", "معادلات", "مندليف", "اتزان", "كهربية"],
        "رياض": ["رياض", "تفاضل", "تكامل", "جبر", "هندسة", "استاتيكا", "ديناميكا", "math", "المعاصر", "100%"],
        "جيولوجيا": ["جيولوجيا", "geology", "علوم بيئة", "صخور", "معادن", "فوالق", "طيات"]
    }
    
    for key, words in keywords.items():
        if key in sub:
            return any(w in src for w in words)
    return True

@app.get("/")
def home():
    return {"status": "online", "message": "Thanwya Smart RAG Engine is Active!"}

@app.post("/api/explain")
async def explain_lesson(req: StudyRequest):
    if not req.api_key:
        raise HTTPException(status_code=400, detail="يرجى توفير API Key صحيح.")
    
    client = genai.Client(
        api_key=req.api_key,
        http_options=types.HttpOptions(headers={"x-goog-api-key": req.api_key})
    )
    
    search_query = req.user_query.strip() if req.user_query.strip() else f"درس {req.lesson}"
    search_text = f"مادة {req.subject} {search_query}"
    
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
                    if is_source_relevant(req.subject, source_file):
                        has_matching_docs = True
                        retrieved_context += f"\n--- [من مرجع: {source_file}] ---\n{doc}\n"
                        if source_file not in sources_used:
                            sources_used.append(source_file)
        except Exception as e:
            print(f"⚠️ تنبيه استعلام ChromaDB: {e}")

    track_name = "علمي علوم" if req.track == "science" else "علمي رياضة"

    # 1️⃣ وضع المحادثة المباشرة والمرنة والرد على استفسار الطالب (Chat Mode)
    if req.user_query.strip():
        prompt = f"""
أنت أستاذ ومعلم خبير أول في المنهج المصري للثانوية العامة لمادة ({req.subject}) لمسار ({track_name}).
أنت الآن في محادثة مباشرة مع طالب في درس "{req.lesson}".

رسالة / سؤال الطالب:
"{req.user_query}"

المراجع المستخرجة من مذكراتكم المرفوعة:
{retrieved_context if has_matching_docs else "لا توجد فقرة نصية مباشرة من المذكرات، اعتمد على كتاب الوزارة الرسمي والمعايير الأكاديمية الدقيقة."}

[قواعد الإجابة والمحادثة الذكية بدون أخطاء]:
1. ركّز تركيزاً تاماً على ما يطلبه ويسأل عنه الطالب بدقة؛ أجب إجابة مباشرة ومرنة وتفاعلية تشفي تساؤله دون إعادة سرد الدرس بالكامل.
2. الدقة العلمية واللغوية الصارمة (Zero Errors):
   - في اللغة العربية والنحو: حلل التركيب وفق مبدأ "المعنى فرع الإعراب"، حدد أركان الجملة الأساسية أولاً، واستوفِ شروط القواعد والضمائر والفضلات بدقة.
   - في الفيزياء والرياضيات: اكتب القانون الرياضي أولاً، عوض بالخطوات العددية الدقيقة خطوة بخطوة، واضبط الإشارات (كإشارات مسارات كيرشوف) والوحدات.
   - في الأحياء والكيمياء والجيولوجيا: التزم بالمصطلحات والتفسيرات المعتمدة بكتاب وزارة التربية والتعليم المصرية.
3. إذا كان الطالب يسأل عن سبب خطأ في إجابة أو يريد توضيحاً أسهل، اشرح له المنطق الرياضي/اللغوي مع ضرب مثال توضيحي جديد وسهل.
4. الأسلوب: معلم تربوي ذكي، مباشر، مشجع، يفهم قصد الطالب ولا يكرر قوالب جاهزة.
"""

    # 2️⃣ وضع الشرح الكامل الشامل للدرس لأول مرة (Full Lesson Mode)
    else:
        prompt = f"""
أنت أستاذ وموجه أول تربوي خبير في المنهج المصري للثانوية العامة لمادة ({req.subject}) لمسار ({track_name}).
مهمتك الشرح التعليمي الشامل والتأسيسي المحكم لدرس "{req.lesson}" في مادة "{req.subject}".

المراجع المستخرجة من مذكراتكم:
{retrieved_context if has_matching_docs else f"⚠️ لم يتم العثور على مذكرات مرفوعة لمادة ({req.subject})، يتم الشرح اعتماداً على كتاب الوزارة الرسمي."}

[معايير الشرح الأكاديمي الدقيق والحديث]:
1. الدقة الأكاديمية: اشرح المفاهيم والقوانين بالتفصيل والتأسيس التراكمي وتفكيك الرموز، مع تجنب أي معلومات خاطئة أو غير دقيقة.
2. اعتمد على المراجع المرفوعة إذا كانت تخص المادة، وأكمل الشرح بمعرفتك التامة بالمنهج المصري.
3. ركز على التريكات، ونقاط الخداع في الامتحانات، والفروق الدقيقة.
4. اختتم الشرح بقسم "مفكرة التلخيص السريع".

الهيكل الإجباري للشرح:
### شرح درس: {req.lesson} ({req.subject})
1. التمهيد والتأسيس التراكمي المطلوب.
2. الشرح والتفكيك التفصيلي للقواعد والمفاهيم والقوانين مع الأمثلة.
3. ⚠️ تريكات وأفكار امتحانات الثانوية العامة.
4. مفكرة التلخيص السريع (نقاط مركزة للحفظ).
"""

    try:
        explanation_text = generate_text_response(prompt, req.api_key, client)
        return {
            "status": "success",
            "explanation": explanation_text,
            "sources": sources_used if has_matching_docs else ["منهج وزارة التربية والتعليم الرسمي"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ أثناء التوليد: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
