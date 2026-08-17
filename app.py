import os
import time
import requests
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# مفتاح الـ API الخاص بك
API_KEY = "AQ.Ab8RN6JQHAEwyKgQ_WkZ3q9ANVnMefxeM49ubpCkZSu5hng4sA"

PDF_FOLDER = "./biology_pdfs"
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

chroma_client = chromadb.PersistentClient(path="./biology_db")
collection = chroma_client.get_or_create_collection(name="biology_materials")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
if not pdf_files:
    print("⚠️ لا توجد ملفات PDF داخل المجلد!")
    exit()

print(f"📚 جاري معالجة {len(pdf_files)} ملف(ات) بنظام الـ Batch السريع...\n")

def get_batch_embeddings(chunks, api_key):
    """إرسال 10 فقرات في طلب واحد لتقليل استهلاك الـ Rate Limit بنسبة 90%"""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    requests_list = [
        {
            "model": "models/gemini-embedding-2",
            "content": {"parts": [{"text": c}]}
        }
        for c in chunks
    ]
    payload = {"requests": requests_list}
    
    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return [item["values"] for item in data["embeddings"]]
            elif resp.status_code == 429:
                print("\n⚠️ تم الوصول للحد الدقيق (429). ننتظر 60 ثانية لفتح الحصة الجديدة...")
                for sec in range(60, 0, -15):
                    print(f"⏳ متبقي {sec} ثانية...")
                    time.sleep(15)
                print("🔄 جاري استئناف الرفع الآن...")
            else:
                print(f"❌ خطأ غير متوقع HTTP {resp.status_code}: {resp.text}")
                time.sleep(5)
        except Exception as e:
            print(f"⚠️ خطأ في الاتصال: {e}، إعادة المحاولة بعد 10 ثوان...")
            time.sleep(10)

BATCH_SIZE = 10  # إرسال 10 فقرات دفعة واحدة

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    print(f"📄 جاري قراءة الملف: {pdf_file}...")
    
    try:
        reader = PdfReader(pdf_path)
        raw_text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                raw_text += extracted + "\n"
                
        chunks = text_splitter.split_text(raw_text)
        if not chunks:
            print(f"⚠️ الملف {pdf_file} مصور (Scan) أو فارغ.\n")
            continue

        new_chunks = []
        new_ids = []
        for j, chunk in enumerate(chunks):
            chunk_id = f"{pdf_file}_chunk_{j}"
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)

        if not new_chunks:
            print(f"⚡ الملف {pdf_file} مرفوع بالكامل سابقاً. تم تخطيه! 🎯\n")
            continue

        print(f"✂️ سيتم رفع {len(new_chunks)} فقرة بنظام الـ Batch...")
        
        for i in range(0, len(new_chunks), BATCH_SIZE):
            batch_chunks = new_chunks[i : i + BATCH_SIZE]
            batch_ids = new_ids[i : i + BATCH_SIZE]
            batch_metadatas = [{"source": pdf_file, "subject": "General"} for _ in range(len(batch_chunks))]

            # جلب Embeddings لـ 10 فقرات بطلب واحد فقط!
            batch_embeddings = get_batch_embeddings(batch_chunks, API_KEY)

            if len(batch_embeddings) == len(batch_chunks):
                collection.upsert(
                    documents=batch_chunks,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                print(f"  └─ تم رفع دفعة {min(i + BATCH_SIZE, len(new_chunks))}/{len(new_chunks)} بنجاح.")
                time.sleep(5)  # مسافة أمان هادئة بين الدفعات

        print(f"✅ تم معالجة {pdf_file} بالكامل بنجاح!\n")
            
    except Exception as e:
        print(f"❌ حدث خطأ أثناء معالجة الملف {pdf_file}: {e}\n")

print("\n🎉 تم تحديث قاعدة البيانات بجميع الكتب والملفات بنجاح تام وبأعلى سرعة!")
