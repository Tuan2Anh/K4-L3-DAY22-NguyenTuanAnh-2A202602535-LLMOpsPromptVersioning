Chuẩn bị repo và môi trường
Về bài lab này
Xây RAG pipeline có LangSmith tracing, quản lý 2 phiên bản prompt trên Prompt Hub, đo chất lượng bằng RAGAS và chặn PII/JSON lỗi bằng Guardrails AI.
Bạn làm được gì sau bài này
Có ít nhất 100 traces trên LangSmith: 50 từ RAG pipeline và 50 từ A/B routing.
Hai prompt V1/V2 được push lên Prompt Hub và được pull về khi chạy.
Có file ragas_report.json với 4 chỉ số RAGAS cho cả V1 và V2, faithfulness ≥ 0.8 ở ít nhất một phiên bản.
Có 2 validator tự viết: che được email, số điện thoại, SSN, số thẻ và sửa được JSON lỗi.
Cần chuẩn bị
Python 3.10 trở lên.
API key của một LLM provider: OpenAI, Gemini, Anthropic, OpenRouter, hoặc Ollama chạy local.
Tài khoản LangSmith miễn phí tại smith.langchain.com.
Git và terminal (Windows: dùng Git Bash).
Trình soạn thảo code, ví dụ VS Code.
Lỗi thường gặp
import ragas báo No module named 'langchain_community.chat_models.vertexai' → chạy pip install "langchain-community<0.4".
config.py báo Thiếu LANGCHAIN_API_KEY → tên biến trong .env phải là LANGCHAIN_API_KEY, không phải LANGSMITH_API_KEY.
UnicodeEncodeError khi chạy lệnh có | tee trên Windows → chạy export PYTHONUTF8=1 trong Git Bash trước.
Bước 4 in Đã redact nhưng Output giống hệt Input → validator đang trả về PassResult; phải trả về FailResult(fix_value=...).
Log Bước 2 báo dùng prompt local (fallback) → pull từ Hub thất bại, kiểm tra lại LANGCHAIN_API_KEY.
Trong lab này, bạn biến một hệ thống hỏi đáp RAG thành một hệ thống quan sát được, so sánh được và có rào chắn. Bạn bắt đầu bằng một RAG pipeline trên FAISS, gắn LangSmith tracing để nhìn thấy từng lần gọi. Sau đó bạn tách prompt ra khỏi code bằng Prompt Hub và chia câu hỏi cho 2 phiên bản prompt. Tiếp theo bạn đo xem phiên bản nào tốt hơn bằng RAGAS. Cuối cùng bạn tự viết validator để chặn thông tin cá nhân và sửa JSON lỗi trong đầu ra của LLM.

Lab gồm 4 nhiệm vụ, mỗi nhiệm vụ 25 điểm, tổng 100 điểm. Thời gian ước tính khoảng 3–4 giờ, trong đó riêng phần chạy RAGAS mất 15–30 phút.

Bài cá nhân
Mỗi học viên tự làm và tự nộp một repo riêng. Deadline: 23:59 ngày 08/10/2026 (GMT+7), trừ khi key coach thông báo khác trong vòng 48 giờ sau buổi lab.

4 nhiệm vụ nối tiếp nhau
1. RAG + tracing

2. Prompt Hub + A/B

3. RAGAS so sánh V1/V2

4. Guardrails

Nhiệm vụ 2 dùng lại vector store của nhiệm vụ 1, và nhiệm vụ 3 dùng lại đúng 2 system prompt của nhiệm vụ 2. Vì vậy hãy làm theo thứ tự. Nhiệm vụ 4 độc lập, không gọi LLM.

Mục tiêu của phần này: chạy python config.py và thấy dòng ✅ Config OK. Đây là dấu hiệu duy nhất cho biết thư viện, API key và tên project LangSmith đã sẵn sàng. Nếu bỏ qua bước này, lỗi cấu hình sẽ chỉ lộ ra giữa chừng các nhiệm vụ sau, khi đó khó đoán nguyên nhân hơn nhiều.

Lấy starter repo
Starter repo của lab:

Starter Day 22
Starter (main)
Starter Day 22
git clone https://github.com/VinUni-AI20k/K4-L3-Track2-Day22-LLMOps-Prompt-Versioning.git

Repo có 4 file bạn sẽ điền, nằm trong src/: 01_langsmith_rag_pipeline.py, 02_prompt_hub_ab_routing.py, 03_ragas_evaluation.py, 04_guardrails_validator.py. Các file config.py, qa_pairs.py và thư mục utils/ đã viết sẵn, bạn chỉ cần gọi chúng.

Cài thư viện
Tạo virtual environment để thư viện của lab không xung đột với dự án khác. Chọn đúng tab theo terminal bạn dùng:

Tạo venv và cài thư viện
Git Bash / macOS / Linux
PowerShell
python -m venv venv
source venv/bin/activate          # Git Bash trên Windows: source venv/Scripts/activate
pip install -r requirements.txt
pip install "langchain-community<0.4"
Chép
Lần cài đầu mất 5–10 phút. Dòng cuối là bắt buộc: nếu pip cài langchain-community bản 0.4, lệnh import ragas ở nhiệm vụ 3 sẽ báo lỗi No module named 'langchain_community.chat_models.vertexai'.

Cấu hình .env
Trong lúc chờ cài, vào smith.langchain.com → Settings → API Keys → Create API Key và sao chép key (bắt đầu bằng lsv2_). Sau đó chạy cp .env.example .env và điền các biến tối thiểu:

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=day22-lab
PROVIDER=openai
OPENAI_API_KEY=sk-...
Chép
Tên biến phải giữ đúng như .env.example (LANGCHAIN_...), vì src/config.py chỉ đọc các tên này. Nếu dùng anthropic hoặc openrouter, vẫn cần OPENAI_API_KEY vì hai provider này lấy embeddings từ OpenAI.

Windows: bật UTF-8 trước khi lưu log
Log của lab có emoji. Khi lưu log ra file bằng | tee, Python trên Windows sẽ lỗi UnicodeEncodeError nếu chưa bật UTF-8. Mỗi lần mở terminal mới, chạy export PYTHONUTF8=1 trong Git Bash (PowerShell: $env:PYTHONUTF8=1).

Kiểm tra
cd src && python config.py
Chép
Bạn cần thấy ✅ Config OK | Provider: OPENAI | Project: day22-lab. Nếu thấy ⚠️ Thiếu biến môi trường, đọc tên biến được liệt kê và sửa .env.

Trước khi sang nhiệm vụ 10/3

`python config.py` in ra `✅ Config OK`.

`git status` không liệt kê file `.env`.

Đã đăng nhập được vào LangSmith dashboard.


Nhiệm vụ 1 — RAG pipeline và LangSmith tracing (25đ)
Mục tiêu: chạy 50 câu hỏi qua RAG pipeline và thấy ít nhất 50 traces rag-query trên LangSmith, mỗi trace chứa câu hỏi, context được truy xuất và câu trả lời.

Bài toán của bước này
Một hệ thống RAG trả lời câu hỏi bằng cách tìm vài đoạn văn liên quan trong kho tài liệu rồi đưa chúng cho LLM làm ngữ cảnh. Khi câu trả lời sai, bạn cần biết lỗi nằm ở đâu: truy xuất sai đoạn, prompt thiếu chỉ dẫn, hay LLM bịa thêm. Nếu chỉ nhìn output trên terminal, bạn không trả lời được câu hỏi đó. LangSmith giải quyết việc này bằng cách ghi lại toàn bộ cây xử lý của mỗi lần gọi, gọi là một trace.

Kho tài liệu của lab là data/knowledge_base.txt (về ML, NLP, LLM, RAG, LangChain, LangSmith, RAGAS và Guardrails). 50 câu hỏi nằm sẵn trong src/qa_pairs.py, biến SAMPLE_QUESTIONS.

Các khái niệm cần nắm
Chunk: đoạn văn nhỏ cắt ra từ kho tài liệu. Lab dùng chunk_size=500 ký tự, chunk_overlap=50; với kho tài liệu này bạn sẽ được khoảng 107 chunks.
FAISS vector store: chỉ mục cho phép tìm chunk gần nghĩa nhất với câu hỏi dựa trên embedding.
Retriever: đối tượng lấy k chunk gần nhất cho một câu hỏi. Lab dùng k=3.
LCEL chain: cách nối các bước bằng toán tử |: dữ liệu đi qua retriever → prompt → LLM → parser.
@traceable: decorator của LangSmith. Mỗi lần hàm được gọi sẽ tạo một trace riêng; các bước LangChain bên trong tự động thành run con của trace đó.
Vai trò của từng hàm trong file
Mở src/01_langsmith_rag_pipeline.py và đọc hết file trước khi viết. Bạn điền 4 phần:

Hàm / biến	Trách nhiệm	Trả về
setup_vectorstore()	Đọc kho tài liệu, chia chunk, tạo FAISS index	vector store
RAG_PROMPT	System message có chỗ trống {context} và human message {question}	ChatPromptTemplate
build_rag_chain()	Tạo retriever k=3 và chain retriever → prompt → LLM → parser	(chain, retriever)
ask()	Gọi chain.invoke(question), được gắn @traceable	câu trả lời dạng chuỗi
Điểm cần giữ: build_rag_chain() phải trả về cả hai giá trị (chain, retriever), vì main() đã viết sẵn dòng chain, retriever = .... {context} phải có trong system prompt, nếu không retriever vẫn chạy nhưng LLM không nhìn thấy tài liệu nào.

Làm từng bước
1. Trong setup_vectorstore(), gọi lần lượt get_embeddings(), load_knowledge_base(), split_text(...) và build_vectorstore(...). Các hàm này đã có trong utils/. Thứ tự quan trọng vì mỗi bước cần output của bước trước.
2. Tạo RAG_PROMPT bằng ChatPromptTemplate.from_messages, theo đúng nội dung gợi ý trong comment TODO của file.
3. Trong build_rag_chain(), tạo retriever, viết format_docs để ghép page_content của các docs bằng "\n\n", rồi nối chain bằng |.
4. Gắn @traceable(name="rag-query", tags=["rag", "step1"]) ngay trên dòng def ask, không có dòng nào chen giữa. Thân hàm chỉ cần trả về chain.invoke(question).
5. Điền 3 dòng còn thiếu trong main() rồi chạy python 01_langsmith_rag_pipeline.py. Với gpt-4o-mini, lượt chạy mất khoảng 2 phút.
Kiểm tra kết quả
Terminal phải in đủ [01/50] đến [50/50], mỗi câu kèm một câu trả lời ngắn. Tuy nhiên, script luôn in thông báo hoàn thành, kể cả khi API key LangSmith sai. Vì vậy bằng chứng thật nằm trên dashboard: mở project của bạn trên smith.langchain.com, đếm được ít nhất 50 traces rag-query. Mở một trace bất kỳ, bạn phải thấy run VectorStoreRetriever trả về 3 documents, sau đó là ChatPromptTemplate, ChatOpenAI và StrOutputParser.

Nếu trace thiếu phần retriever, chain của bạn đang không đi qua retriever. Rubric trừ 3 điểm cho lỗi này. Khi đã thấy đủ cây xử lý, chụp màn hình danh sách traces và lưu thành evidence/01_langsmith_traces.png. Retriever và cách ghép context ở đây sẽ được dùng lại nguyên vẹn ở nhiệm vụ 2.

Nhiệm vụ 2 — Prompt Hub và A/B routing (25đ)
Mục tiêu: 2 prompt V1/V2 xuất hiện trên Prompt Hub, được pull về khi chạy, và 50 câu hỏi được chia cho V1/V2 theo một quy tắc tất định, có nhãn phiên bản trong log.

Bài toán của bước này
Ở nhiệm vụ 1, prompt nằm cứng trong code. Muốn thử một cách diễn đạt khác, bạn phải sửa code và chạy lại, và không có lịch sử nào cho biết phiên bản nào từng chạy. Prompt Hub của LangSmith lưu prompt như một tài nguyên có phiên bản: code chỉ cần biết tên prompt và pull về khi chạy.

Khi đã có 2 phiên bản, bạn cần chia lưu lượng cho chúng để so sánh, gọi là A/B routing. Quy tắc chia phải tất định: cùng một request_id luôn đi vào cùng một phiên bản. Nếu chia ngẫu nhiên, chạy lại hai lần sẽ ra hai cách chia khác nhau và không thể đối chiếu kết quả. Rubric trừ 5 điểm nếu routing ngẫu nhiên.

Các khái niệm cần nắm
push_prompt / pull_prompt: hai hàm của langsmith.Client để đẩy prompt lên Hub và lấy về theo tên.
Fallback local: file đã viết sẵn nhánh except. Nếu pull thất bại, chương trình dùng prompt local để không dừng lại. Nhánh này giúp chương trình chạy tiếp, nhưng nếu nó xảy ra thì bạn mất điểm tiêu chí 2.2/2.3 vì prompt không đến từ Hub.
Hash MD5 của request_id: biến một chuỗi như req-0003 thành một số nguyên cố định. Số chẵn → V1, số lẻ → V2.
Thiết kế trong file
Mở src/02_prompt_hub_ab_routing.py. PROMPT_V1 và PROMPT_V2 được tạo ngay khi import file, từ SYSTEM_V1 và SYSTEM_V2. Vì vậy nếu chưa điền 2 biến này, file sẽ lỗi Invalid template: Ellipsis trước khi chạy tới main().

Hàm get_prompt_version() trả về tên prompt (PROMPT_V1_NAME hoặc PROMPT_V2_NAME), không phải chuỗi "v1", vì main() dùng kết quả này để tra trong dict prompts. Hàm ask_ab() tự retrieve, ghép context rồi chạy prompt | llm | StrOutputParser(), và được gắn @traceable(name="ab-rag-query", ...) để tạo thêm 50 traces.

Gợi ý trong file thiếu {context}
Comment gợi ý cho SYSTEM_V1/SYSTEM_V2 không có {context}. Chép nguyên gợi ý thì LLM không nhận tài liệu mà chương trình vẫn chạy, không báo lỗi gì. Hãy kết thúc mỗi system prompt bằng \n\nContext:\n{context}.

Làm từng bước
1. Đổi PROMPT_V1_NAME và PROMPT_V2_NAME thành tên riêng của bạn, ví dụ nguyen-van-a-rag-prompt-v1, để không trùng với người khác.
2. Viết SYSTEM_V1 (ngắn gọn, 2–4 câu) và SYSTEM_V2 (có cấu trúc, giọng chuyên gia, 3–5 câu). Hai prompt phải khác nhau về ngữ nghĩa, không chỉ khác vài từ. Cả hai đều kết thúc bằng phần Context:\n{context}.
3. Điền push_prompts_to_hub và pull_prompts_from_hub theo dòng gợi ý trong comment của từng TODO.
4. Viết get_prompt_version() bằng MD5, rồi viết ask_ab() và các dòng còn thiếu trong main().
5. Chạy và lưu log: python 02_prompt_hub_ab_routing.py | tee ../evidence/02_ab_routing_log.txt.
Kiểm tra kết quả
Phần đầu log phải có ✅ Đã push V1 → https://smith.langchain.com/prompts/... và ↓ Đã pull '...' từ Hub cho cả hai prompt. Nếu thấy dòng báo dùng prompt local (fallback), pull đã thất bại. Thường là do API key sai; sửa .env rồi chạy lại. Mỗi câu hỏi trong log có nhãn [prompt-v1] hoặc [prompt-v2], và dòng tổng kết 📊 Routing: V1=... | V2=... cho thấy cả hai phiên bản đều nhận câu hỏi. Tỉ lệ không cần 25/25; với req-0000 đến req-0049, hash cho khoảng 19/31.

Chạy lại lần thứ hai có thể thấy 409 Conflict ... Nothing to commit khi push. Đây không phải lỗi: prompt chưa đổi nên Hub không tạo phiên bản mới. Lần chạy lại cũng cho đúng nhãn cũ cho từng câu, đó chính là tính tất định.

Cuối cùng vào LangSmith → Prompt Hub, chụp màn hình 2 prompt của bạn và lưu thành evidence/02_prompt_hub.png. Tổng số traces trong project bây giờ phải là ít nhất 100.

Nhiệm vụ 3 — Đánh giá V1 và V2 bằng RAGAS (25đ)
Mục tiêu: chạy 50 cặp hỏi–đáp qua cả V1 và V2, chấm bằng 4 chỉ số RAGAS và lưu data/ragas_report.json, với faithfulness ≥ 0.8 ở ít nhất một phiên bản.

Bắt đầu sớm
Phần chạy mất 15–30 phút vì RAGAS gọi LLM nhiều lần cho mỗi mẫu. Lần chạy thử với gpt-4o-mini mất khoảng 21 phút. Bắt đầu ngay khi xong nhiệm vụ 2 và không đóng terminal.

Bài toán của bước này
Ở nhiệm vụ 2 bạn đã có 2 phiên bản prompt, nhưng chưa biết phiên bản nào tốt hơn. Đọc vài câu trả lời rồi đoán thì không đủ cơ sở. RAGAS chấm hệ thống RAG bằng số, dựa trên câu hỏi, câu trả lời, các đoạn context đã truy xuất và đáp án chuẩn. Đáp án chuẩn nằm trong QA_PAIRS của src/qa_pairs.py: một list các dict có khóa question và reference.

Bốn chỉ số
Chỉ số	Đo điều gì
faithfulness	Câu trả lời có bám vào context đã truy xuất, hay bịa thêm
answer_relevancy	Câu trả lời có đúng trọng tâm câu hỏi không
context_recall	Context truy xuất có chứa đủ thông tin của đáp án chuẩn không
context_precision	Các đoạn context truy xuất có thực sự liên quan không
faithfulness là chỉ số rubric dùng làm ngưỡng (≥ 0.8). Nó tụt mạnh khi LLM không nhận được context, vì vậy {context} trong prompt ở nhiệm vụ 2 quan trọng như vậy.

Thiết kế trong file
Mở src/03_ragas_evaluation.py. Có 3 điểm cần giữ đúng.

Thứ nhất, SYSTEM_V1/SYSTEM_V2 phải giống hệt nhiệm vụ 2, kể cả {context}; nếu khác thì kết quả không còn là đánh giá của 2 prompt đã chạy trên Hub.

Thứ hai, trong run_rag(), contexts phải là list[str], mỗi phần tử là một đoạn. Chuỗi ghép ctx_str chỉ dùng để đưa vào prompt. RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

Thứ ba, mỗi SingleTurnSample cần đúng 4 trường: user_input, response, retrieved_contexts, reference. Phần tính trung bình điểm, in bảng so sánh và tạo dict report đã viết sẵn.

Làm từng bước
1. Copy SYSTEM_V1 và SYSTEM_V2 từ file nhiệm vụ 2.
2. Viết run_rag(): retrieve, tạo contexts dạng list, ghép ctx_str, chạy chain, trả về {"answer": ..., "contexts": ...}.
3. Điền collect_rag_outputs() và build_ragas_dataset() theo đúng tên khóa trong comment TODO.
4. Trong run_ragas_eval(), gọi evaluate(dataset, metrics=[...4 chỉ số...], llm=llm_eval, embeddings=emb_eval). Trong main(), tạo vector store và ghi report ra file bằng report_path.write_text(json.dumps(report, indent=2), encoding="utf-8").
5. Chạy python 03_ragas_evaluation.py, rồi cp ../data/ragas_report.json ../evidence/03_ragas_report.json.
Kiểm tra kết quả
Khi chạy xong, terminal in bảng Metric | V1 | V2 | Winner và dòng ✅ Đạt mục tiêu: faithfulness = ... ≥ 0.8. Lần chạy thử đạt faithfulness 0.955 (V1) và 0.950 (V2); kết quả của bạn sẽ khác tùy prompt và model. Chụp màn hình bảng này và lưu thành evidence/03_ragas_scores.png. Chạy python -m json.tool ../evidence/03_ragas_report.json để chắc file là JSON hợp lệ và có cả prompt_v1_scores lẫn prompt_v2_scores.

Nhiều dòng LLM returned 1 generations instead of requested 3 trong log là bình thường. Nếu faithfulness dưới 0.8 ở cả hai phiên bản, kiểm tra {context} trong prompt trước, sau đó mới thử giảm chunk_size hoặc tăng k. Viết vài câu giải thích vì sao V1 hoặc V2 cao hơn sẽ được điểm thưởng.

Nhiệm vụ 4 — Guardrails AI validators (25đ)
Mục tiêu: tự viết PIIDetector che được ít nhất 3 loại thông tin cá nhân và JSONFormatter tự sửa JSON lỗi, demo trên các test case có sẵn trong file.

Bài toán của bước này
Ngay cả khi RAG trả lời đúng, đầu ra của LLM vẫn có thể lộ thông tin cá nhân hoặc sai định dạng mà hệ thống phía sau cần. Guardrails AI chen một lớp kiểm tra giữa LLM và người dùng: mỗi validator kiểm tra một điều kiện, và on_fail quyết định làm gì khi điều kiện không đạt. Lab yêu cầu OnFailAction.FIX, tức là thay đầu ra bằng một phiên bản đã sửa thay vì báo lỗi. Bước này không gọi LLM; bạn kiểm tra validator trên các chuỗi mẫu có sẵn.

Điểm dễ sai nhất
Với on_fail=OnFailAction.FIX, Guardrails 0.11 chỉ thay output bằng fix_value của FailResult. Docstring và comment TODO trong file đang gợi ý PassResult(value_override=...). Làm theo gợi ý đó thì log vẫn in ⚠️ Đã redact, nhưng dòng Output: giống hệt Input:, nghĩa là PII không hề bị che. Hãy dùng quy ước sau:

Tình huống	Trả về
Không có PII / JSON đã hợp lệ	PassResult()
Có PII / JSON sửa được	FailResult(error_message=..., fix_value=<bản đã sửa>)
JSON không thể sửa	FailResult(error_message=..., fix_value=<JSON dự phòng>)
Hàng cuối ứng với tiêu chí 4.7 của rubric: khi không sửa được, phải trả về một JSON dự phòng như {"error": "...", "raw": "..."} chứ không để output là None.

on_fail phải truyền vào constructor của validator: Guard().use(PIIDetector(on_fail=OnFailAction.FIX)). Truyền vào Guard.use() sẽ bị trừ 3 điểm. Rubric cũng trừ 5 điểm nếu dùng validator có sẵn từ Guardrails Hub thay vì tự viết.

Làm từng bước
1. Trong PIIDetector.validate(), duyệt self.PII_PATTERNS (regex đã cho sẵn), dùng re.findall để tìm match và thay mỗi match bằng [LOẠI_REDACTED] trong redacted_text.
2. Trả về FailResult(..., fix_value=redacted_text) khi tìm thấy PII, PassResult() khi không có.
3. Trong JSONFormatter._repair(), thay nháy đơn bằng nháy đôi và xóa dấu phẩy thừa trước } hoặc ] (phần gỡ markdown fences đã có sẵn).
4. Trong JSONFormatter.validate(), viết 3 nhánh theo bảng ở trên, sau đó điền hai dòng Guard().use(...) và guard.validate(text) trong hai hàm demo.
5. Chạy và lưu log ra cả hai file evidence trong một lần: python 04_guardrails_validator.py | tee ../evidence/04_pii_demo_log.txt ../evidence/04_json_demo_log.txt.
Kiểm tra kết quả
Phần PII có 6 test case. Dòng Output: của 5 case đầu phải chứa [EMAIL_REDACTED], [PHONE_REDACTED], [SSN_REDACTED] hoặc [CREDIT_CARD_REDACTED]; case Clean giữ nguyên. Nếu Output giống hệt Input, bạn vẫn đang trả về PassResult. Với regex có sẵn, số (555) 867-5309 có thể được che thành ([PHONE_REDACTED], sót dấu (; phần số vẫn được che.

Phần JSON có 5 test case: 3 case lỗi (markdown fences, nháy đơn, dấu phẩy thừa) phải in ra JSON đã format lại, case Truly invalid phải in ra JSON dự phòng. Các cảnh báo opentelemetry ... Failed to export spans là telemetry của Guardrails, có thể bỏ qua.

Kiểm tra và nộp bài
Mục tiêu: nộp một repo public, đặt tên đúng quy ước, chứa code đã hoàn thành và đủ 7 file evidence, kèm link LangSmith project, trước deadline.

Chạy lại toàn bộ
Trước khi đóng gói, chạy cd src && python run_all.py để chắc 4 bước đều chạy không lỗi khi đi liền nhau. Nếu chỉ muốn chạy lại một bước, dùng python run_all.py --step 3.

Đặt tên repo
Tạo repo public trên GitHub cá nhân với tên:

K4-L3-DAY22-HoVaTen-MSSV-LLMOpsPromptVersioning
Chép
Ví dụ: K4-L3-DAY22-NguyenVanA-2A20260000-LLMOpsPromptVersioning. Viết không dấu, không khoảng trắng, các phần ngăn cách bằng -. Repo đặt sai tên có thể không được chấm.

Đủ 7 file evidence
File	Nội dung
evidence/01_langsmith_traces.png	LangSmith với ≥ 50 traces
evidence/02_prompt_hub.png	Prompt Hub có 2 prompt đặt tên riêng
evidence/02_ab_routing_log.txt	Log 50 câu, có nhãn v1/v2
evidence/03_ragas_scores.png	Bảng so sánh V1 vs V2 trên terminal
evidence/03_ragas_report.json	Bản sao data/ragas_report.json
evidence/04_pii_demo_log.txt	Log các test case PII
evidence/04_json_demo_log.txt	Log các test case JSON
Thêm evidence/README.md phân tích ngắn V1 so với V2 sẽ được điểm thưởng.

Không commit API key
Không commit .env và không để API key trong code, log hay ảnh chụp màn hình. Bài có API key trong code bị trừ tự động 10 điểm. Nếu lỡ commit key, thu hồi key đó ngay trên trang của nhà cung cấp.

Nộp ở đâu và chấm thế nào
Nộp qua cổng nộp bài của khóa học 2 đường dẫn: URL GitHub repository và URL LangSmith project (tổng cộng ≥ 100 traces). Bài được chấm theo commit cuối cùng trước deadline, dựa trên RUBRIC.md trong repo: phần bắt buộc tối đa 100 điểm, điểm thưởng cộng tối đa 10 điểm. Nộp muộn bị trừ điểm theo quy định chung của khóa học; chi tiết về sử dụng AI, sao chép và sửa bài sau deadline nằm trong RULES.md.

Trước khi bấm nộp0/5

Repo public, tên đúng quy ước `K4-L3-DAY22-...`.

`git ls-files` không có `.env`.

Đủ 7 file trong `evidence/`, không file nào rỗng.

LangSmith project có ít nhất 100 traces.

Mở repo bằng cửa sổ ẩn danh vẫn thấy đủ file.