
<!--
**junho-create/junho-create** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
## Hi there I'm Junho Yeo👋
🎓 Industrial Engineering student at **Yonsei University**  
💡 Interested in **LLM(VLM), Generative Models**  

---

### 🧩 Projects
#### 🧪 Major Coursework Projects
- Longtailservice_Recommendation_teamproject.zip  
  : Team project for "Advanced Programming (24-2)", developing a long-tail service recommendation system using collaborative filtering and content-based recommendation algorithms
- IIM_TermProject_Team7.zip
  : Term project for "Industrial Data Management (24-2)",**Leukocyte Donation Management System (LDMS)** using Database
- Project_StockPrediction.zip
   : Final project for “Optimization in Artificial Intelligence (25-1)”, predicting stock price trends based on historical financial data using **GARCH**, **LSTM**, and **CatBoost**

#### 💼 25-2 Long-term Internship(Samsung Electronics) ('25.09. ~ 12.)
- **3 projects — code and data are not included due to company security policy** (titles only)
  → Samsung Electronics, Semiconductor Research Institute – Process Development Innovation Team (Sep – Dec 2025)

  => Fault prediction for photomask equipment using SPRT
  
  => CNN-based image classification for fault detection in photomask equipment
  
  => Deep learning-based image classification scoring method using Deep SVDD and SAD for fault detection in photomask equipment

#### 🔬 BISPL winter Internship ('25.12.~'26.03.)
- **ISBI 2026 Low Concentration Reconstruction MPI Challenge**
  : Achieved **Top 5 finalist** and accepted challenge paper; contributed to experiment analysis and presentation materials for the international conference
  
#### 🧠 Academic Club (DSL) Projects ('25.07.~'26.05)
- 25-2_DSL_Medical_team_EDA
  : **CGM × IOB MixedLM analysis** for insulin sensitivity — per-patient sensitivity estimated with **Linear Mixed Model** and **ARIMA**, quantifying the time lag of insulin/carbohydrate effects on blood glucose ('25.07~08)
- 25-2_DSL_Modeling_NLP2_HospitalAgent
  : **Multi-agent hospital reservation system** (LangGraph / MCP / Supabase / RAG) with a multi-turn E2E evaluation suite ('25.08~09)
- 25-2_DSL_companyproject(DSLXLattice)
  : **Industry project** — contract PDF information extraction pipeline (OCR + LLM hybrid) auto-extracting 11 key fields with evidence spans and confidence scores
- 26_1_DSL_Semiconductor_EDA
  : **LithoBench lithography mask optimization EDA** — stage-wise IoU/XOR error decomposition, FFT-based pattern frequency analysis, and ILT cost–efficiency trade-off analysis ('26.01~02)
- 26_1_DSL_Modeling_Multimodal_AIAvatar
  : **Multimodal AI avatar prototype** integrating LLM-based dialogue, STT, TTS, RAG, and real-time lip-sync rendering on LiveKit; evaluated with BLEURT and LLM-as-a-Judge
- 26-1_DSL_companyproject(DSLXLGAIresearch)
  : **Industry project** — time-series forecasting with text signals, using **DoubleAdapt**(with K-of-N Loss Function consensus logic) meta-learning for online adaptation to distribution shift


#### 🏢 26-1 Summer Internship(genon) ('26.06. ~ 08.)
- **Document-parsing VLM pipeline** — end-to-end data curation → fine-tuning → benchmark evaluation for a document OCR/layout model (**Qwen3.5-9B**)

  => Built an automatic dataset curation engine (MinerU2.5-Pro style): cross-model cross-validation of 3 VLMs (target / dots.ocr / PaddleOCR-VL) scored per subtask (**PageIoU** for layout, **edit distance** for text, **TEDS** for tables, **CDM** for formulas) to auto-tier pages into Easy/Medium/Hard and adopt pseudo-labels
  
  => Designed an element-level **rescue** stage (N:M bbox grouping of two external models' outputs) and a **hardcase judge** stage (122B VLM, generate + render-then-verify) to auto-label the hardest pages without human annotation
  
  => Fine-tuned **Qwen3.5-9B with LoRA (r=128)** on a 37,818-sample combined dataset (auto-curated + human-GT reference) using 3–4×H100 / DeepSpeed, running versioned experiments with W&B tracking
  
  => Built an end-to-end evaluation pipeline on **T²-RAGBench**, statistically verifying that our in-house document parser outperforms commercial OCR parsers and competitors in downstream generation accuracy (**48.44% vs 46.51%, p=0.006**)



  

  


---

📫 Contact: [asap03153@yonsei.ac.kr] 
