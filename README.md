# Medical Pharmacy Assistant — Streamlit App

واجهة Streamlit للمشروع، بتستخدم نفس الـ`retrieval.py` و`rag.py` بتوع
النوت بوك (Cohere rerank + Gemini)، من غير أي تكرار للـlogic.

## الملفات

```
app.py                          # واجهة Streamlit
retrieval.py                    # retrieve_chunks() -- Weaviate hybrid search (Rania)
rag.py                          # rerank + prompt + LLM + citations (Cohere rerank)
requirements.txt                # dependencies
.streamlit/secrets.toml.example # نموذج الـsecrets
```

## التشغيل محليًا

1. ثبّتي المكتبات:
   ```bash
   pip install -r requirements.txt
   ```

2. اعملي نسخة من ملف الـsecrets وحطي فيها القيم الحقيقية:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   وافتحي `.streamlit/secrets.toml` وحطي:
   - `WEAVIATE_URL` و `WEAVIATE_API_KEY` — نفس القيم اللي في Colab Secrets
   - `COHERE_API_KEY` — مفتاح Cohere
   - `GOOGLE_API_KEY` — مفتاح Gemini (من Google AI Studio)

3. شغّلي الـapp:
   ```bash
   streamlit run app.py
   ```

## النشر على Streamlit Community Cloud

1. ارفعي الفولدر ده على GitHub repo (**من غير** ملف `secrets.toml` نفسه —
   خليه في `.gitignore`، الـ`.example` بس هو اللي يترفع).
2. من [share.streamlit.io](https://share.streamlit.io) اختاري الـrepo،
   وحطي `app.py` كملف رئيسي.
3. من **App settings → Secrets** الصقي نفس محتوى `secrets.toml.example`
   بس بالقيم الحقيقية.
4. Deploy.

## ملاحظات

- الموديل المستخدم للـembeddings (`BAAI/bge-base-en-v1.5`) بيتحمّل مرة واحدة
  بس بفضل `st.cache_resource` — مش هيعيد التحميل مع كل سؤال.
- الـ"I don't know based on the provided medical sources." بتتحسب **قبل**
  ما تتنادى الـLLM، فمفيش تكلفة إضافية على أسئلة خارج النطاق.
- في الشريط الجانبي (Advanced settings) تقدري تتحكمي في عدد المصادر
  (`top_k`) وعتبة الـrelevance (`min_rerank_score`) بدون ما تلمسي الكود.
