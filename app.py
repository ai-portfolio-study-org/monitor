import streamlit as st
import json
import glob
import pandas as pd
import os
import subprocess
import tempfile





st.set_page_config(page_title="모델 성능 모니터링 대시보드", layout="wide")

st.title("📈 모델 성능 비교 대시보드")

st.sidebar.header("🎙️ 음성 파일 업로드 (mp3)")
# 파일 업로드 기능
st.sidebar.header("📤 새 모델 파일 업로드")

uploaded_audio = st.sidebar.file_uploader("음성 파일 업로드 (.mp3)", type=["mp3"])
audio_process_button = st.sidebar.button("STT 변환 실행")


uploaded_file = st.sidebar.file_uploader("모델 파일 업로드 (.onnx, .gguf, .bin)", type=["onnx", "gguf", "bin"])
model_type = st.sidebar.selectbox("모델 타입 선택", ["STT", "NLU", "인증"])
upload_button = st.sidebar.button("업로드 및 평가")


if uploaded_audio and audio_process_button:
    # 업로드한 mp3 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
        tmp_audio_file.write(uploaded_audio.getbuffer())
        mp3_path = tmp_audio_file.name

    # 변환된 wav 파일 경로
    wav_path = mp3_path.replace(".mp3", ".wav")

    # ffmpeg로 mp3 -> wav 변환
    os.system(f"ffmpeg -i {mp3_path} -ar 16000 -ac 1 {wav_path}")

    # whisper.cpp or STT 엔진 호출해서 wav → 텍스트 변환
    # 여기서는 예시로 dummy 텍스트
    stt_text = "송금 해줘"

    # 변환 결과 저장 (results/stt/폴더에)
    os.makedirs("results/stt", exist_ok=True)
    stt_output_path = os.path.join("results/stt", os.path.basename(wav_path).replace(".wav", ".txt"))
    with open(stt_output_path, "w", encoding="utf-8") as f:
        f.write(stt_text)

    st.sidebar.success(f"✅ STT 변환 완료! 결과 파일: {stt_output_path}")
    st.sidebar.info("NLU 탭에서 변환된 텍스트로 Intent 테스트 해보세요.")

    # 임시 파일 삭제
    os.remove(mp3_path)
    os.remove(wav_path)



if uploaded_file and upload_button:
    # 확장자에 따라 평가 방식 결정
    ext = os.path.splitext(uploaded_file.name)[-1].lower()

    if ext == ".bin":
        model_format = "cpp"
    elif ext == ".onnx":
        model_format = "onnx"
    elif ext == ".gguf":
        model_format = "gguf"
    else:
        model_format = "unknown"

    # 메모리에 올려진 모델 바이너리 저장 없이 평가
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_model_path = tmp_file.name

    # evaluate.py 호출
    subprocess.run([
        "python", "evaluate.py",
        "--model_path", tmp_model_path,
        "--model_type", model_type,
        "--model_format", model_format,
        "--original_name", uploaded_file.name  # 원래 파일 이름도 넘긴다
    ])

    # 평가 끝난 후 temp 파일 삭제
    os.remove(tmp_model_path)

    st.sidebar.success("✅ 모델 평가 완료! (모델 파일은 저장하지 않고 결과만 저장함)")
    st.sidebar.info("좌측 새로고침 후 대시보드에 반영됩니다.")

# ------------------------------
# 기존 대시보드 부분
# ------------------------------

# 탭 구성
stt_tab, nlu_tab, auth_tab = st.tabs(["🎤 STT (Speech-to-Text)", "🧠 NLU (Natural Language Understanding)", "🔒 인증 (Authentication)"])

def load_results(folder_name):
    files = glob.glob(f"results/{folder_name}/*.json")
    results = []
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            results.append(data)
    return results


### STT 탭 ###
### STT 탭 ###
with stt_tab:
    st.header("🎤 STT 모델 성능 비교")

    stt_results = load_results("stt")

    if stt_results:
        df_stt = pd.DataFrame(stt_results)
        df_stt.set_index("ModelName", inplace=True)

        st.subheader("✅ Word Error Rate (WER) 비교")
        st.bar_chart(df_stt["WER"])

        st.subheader("✅ Character Error Rate (CER) 비교")
        st.bar_chart(df_stt["CER"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⏱️ Latency (ms)")
            st.bar_chart(df_stt["Latency(ms)"])
        with col2:
            st.subheader("🚀 Throughput (req/s)")
            st.bar_chart(df_stt["Throughput(req/s)"])
    else:
        st.warning("STT 결과 파일이 없습니다.")


### NLU 탭 ###
### NLU 탭 ###
with nlu_tab:
    st.header("🧠 NLU 모델 성능 비교")

    nlu_results = load_results("nlu")

    if nlu_results:
        df_nlu = pd.DataFrame(nlu_results)
        df_nlu.set_index("ModelName", inplace=True)

        st.subheader("✅ Accuracy 비교")
        st.bar_chart(df_nlu["Accuracy"])

        st.subheader("✅ F1 Score 비교")
        st.bar_chart(df_nlu["F1"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⏱️ Latency (ms)")
            st.bar_chart(df_nlu["Latency(ms)"])
        with col2:
            st.subheader("🚀 Throughput (req/s)")
            st.bar_chart(df_nlu["Throughput(req/s)"])
    else:
        st.warning("NLU 결과 파일이 없습니다.")

    # --- NLU 추가 테스트 기능 (STT 결과 받아오기) ---
    st.divider()
    st.subheader("🧪 STT 결과 기반 NLU 테스트")

    # STT 결과 텍스트 로딩
    stt_output_files = glob.glob("results/stt/*.txt")
    stt_text = ""

    if stt_output_files:
        # 최근 파일 하나 읽기
        latest_stt_file = max(stt_output_files, key=os.path.getctime)
        with open(latest_stt_file, "r", encoding="utf-8") as f:
            stt_text = f.read()
        
        st.text_area("📄 STT 변환 텍스트", value=stt_text, height=150)

        if st.button("NLU 테스트 실행"):
            # 여기서 NLU 모델 호출해야 함
            # 예시로 랜덤 intent 반환
            dummy_intents = ["송금", "계좌조회", "잔액확인", "이체"]
            import random
            detected_intent = random.choice(dummy_intents)
            st.success(f"✅ 분류된 Intent: {detected_intent}")
    else:
        st.info("ℹ️ STT 변환 텍스트 파일이 없습니다. 먼저 STT 추론을 수행하세요.")


### 인증 탭 ###
with auth_tab:
    st.header("🔒 인증 모델 성능 비교")

    auth_results = load_results("auth")

    if auth_results:
        df_auth = pd.DataFrame(auth_results)
        df_auth.set_index("ModelName", inplace=True)

        st.subheader("✅ Equal Error Rate (EER) 비교")
        st.bar_chart(df_auth["EER"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⏱️ Latency (ms)")
            st.bar_chart(df_auth["Latency(ms)"])
        with col2:
            st.subheader("🚀 Throughput (req/s)")
            st.bar_chart(df_auth["Throughput(req/s)"])
    else:
        st.warning("인증 결과 파일이 없습니다.")
