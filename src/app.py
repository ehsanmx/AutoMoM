import streamlit as st
import torch
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from natsort import natsorted
import os

class AutoMoM:
    # TODO move this to config
    model = "llama-2-7b-chat.Q5_K_M.gguf"
    template = """[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    {transcribe}

    Human:{question}
    Assistant:
    [/INST]
    """
    prompt_template = PromptTemplate.from_template(template=template)

    def __init__(self):
        self.main()

    def main(self):        
        self.init_ui()
        st.sidebar.markdown(f"* Loading the model {self.model} is complete.")
        transcribe = self.init_audio_uploader()
        st.markdown(transcribe)
        if transcribe != "":
            self.summarize_transcript(transcribe)
        # self.llm_gen(transcribe=transcribe)

    def init_ui(self):
        with st.sidebar:
            st.title('AutoMoM 1.0')
            st.write('Made with ♥️ by [Ehsan Zanjani](https://www.linkedin.com/in/ezanjani/)')
            st.markdown('## Logs')
            
    
    def init_audio_uploader(self):
        transcribe = ""
        allowed_extensions = ['mp4', 'm4a', 'mp3', 'wav']
        uploaded_file = st.file_uploader("Upload a video or audio file", type=allowed_extensions)
        if uploaded_file is not None:
            output_folder = "output"
            os.makedirs(output_folder, exist_ok=True)
            st.info("Your file is uploaded successfully!", icon='ℹ️')
            # self.split_audio(uploaded_file, output_folder)

            st.success("Audio split successfully. please wait for the result.")

            transcribe = self.transcribe_audio()
            st.session_state.transcribe = transcribe
            
        return transcribe

        
    def split_audio(self, uploaded_file, output_folder, segment_length=60):
        audio_segment = None
        if uploaded_file.name.endswith(".mp4"):
            # Save uploaded video to a temporary file
            temp_video_file = "temp_video.mp4"
            with open(temp_video_file, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Extract audio from video
            video = VideoFileClip(temp_video_file)
            audio = video.audio
            audio_file = "temp_audio.mp3"
            audio.write_audiofile(audio_file)
            audio_segment = AudioSegment.from_mp3(audio_file)
            # Clean up temporary files
            os.remove(temp_video_file)
            os.remove(audio_file)
        elif uploaded_file.name.endswith(".m4a"):
            audio_segment = AudioSegment.from_file(uploaded_file, format="m4a")
        elif uploaded_file.name.endswith(".mp3"):
            audio_segment = AudioSegment.from_mp3(uploaded_file)
        elif uploaded_file.name.endswith(".wav"):
            audio_segment = AudioSegment.from_wav(uploaded_file)
            
        # Split audio into segments of 'segment_length' seconds
        for i in range(0, len(audio_segment), segment_length * 1000):
            part = audio_segment[i:i + segment_length * 1000]
            part.export(f"{output_folder}/minute{i // 1000 // segment_length}.mp3", format="mp3")
    
    def transcribe_audio(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        files = os.listdir("output")
        # Filter out files with the .mp3 extension
        mp3_files = [file for file in files if file.endswith('.mp3')]
        mp3_files = natsorted(mp3_files)
        full_text = ""
        for file in mp3_files:
            result = pipe(f"output/{file}")
            st.write(file)
            st.markdown(result["text"])
            full_text += result["text"]

        return full_text


    def summarize_transcript(self, transcribe):
        summarized_text = self.llm_gen("generate the summary as bullet point list, max 20 items", transcribe)
        return summarized_text

    def generate_action_points(self, transcribe):
        action_points = self.llm_gen("generate the action point list", transcribe)
        return action_points
    
    def llm_gen(self, prompt: str, transcribe :str):
        # llm = Llama(model_path=f"model/{model}", chat_format="llama-2")
        self.llm = LlamaCpp(
            model_path=f"model/{self.model}",
            temperature=0.75,
            max_tokens=4096,
            n_ctx=4096,
            top_p=1,
            streaming=True,            
            callbacks=[StreamingStdOutCallbackHandler()],
            verbose=True,  # Verbose is required to pass to the callback manager
        )

        with st.chat_message("assistant"):
            with st.spinner('Thinking...'):
                full_response = []
                placeholder = st.empty()
                if 'transcribe' in st.session_state:
                    transcribe = st.session_state.transcribe
                print(transcribe)
                formatted_prompt = self.prompt_template.format(transcribe=transcribe, question=prompt)
                for wordstream in self.llm.stream(formatted_prompt):
                    if wordstream:
                        full_response.append(wordstream)
                        result = "".join(full_response).strip()
                        placeholder.markdown(result)

                st.session_state.output_text = "".join(full_response).strip()
                st.toast("Processing complete!", icon='✅')
                st.spinner('Complete')
            # st.session_state.messages.append({"role":"assistant","content": result})
                

# Running the app
if __name__ == "__main__":
   autoMoM = AutoMoM()