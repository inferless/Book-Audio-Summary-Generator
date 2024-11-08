from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pypdf import PdfReader
from TTS.api import TTS
import torch
import io
import base64
import requests

class InferlessPythonModel:
    @staticmethod
    def pdf_to_text(pdf_path):
        reader = PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)

    @staticmethod
    def split_text_into_chunks(text, chunk_size=4000):
        words = text.split(" ")
        for i in range(0, len(words), chunk_size):
            yield " ".join(words[i:i + chunk_size])

    @staticmethod
    def download_book(url, file_name="textbook.pdf"):
        response = requests.get(url)
        with open(file_name, "wb") as file:
            file.write(response.content)
        return file_name

    def initialize(self):
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.llm = LLM(model=model_id, dtype="float16")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.prompts = ["Summarizing the following text from the chapters of a book.",
                       "Write comprehensive notes summarizing the following book. Write in a such a way that it can be read as a speech. Follow this format: Summary of the book <Book-Name>; Introduction; Summarize the chapters in short; Conclusion"
                       ]
        
    def generate_summary(self,prompts_idx,max_tokens,chunk):
        sampling_params = SamplingParams(max_tokens=max_tokens)
        messages = [
                {"role": "system", "content": self.prompts[prompts_idx]},
                {"role": "user", "content": chunk}
            ]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        result = self.llm.generate(input_text, sampling_params)
        summary = [output.outputs[0].text for output in result][0].split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        
        return summary
        
    def infer(self,inputs):
        book_url = inputs['book_url']
        book_name = self.download_book(book_url)
        parsed_text = self.pdf_to_text(book_name)
                
        all_summaries_list = []
        for chunk in self.split_text_into_chunks(parsed_text, chunk_size=4000):
            chunk_summary = self.generate_summary(0,200,chunk)
            all_summaries_list.append(chunk_summary)
        
        all_summaries = "\n\n".join(all_summaries_list)
        final_summary = self.generate_summary(0,1024,all_summaries)
        wav_file = io.BytesIO()
        
        self.tts.tts_to_file(
                text=final_summary,
                file_path=wav_file,
                speaker="Kazuhiko Atallah",
                language="en",
        )    
        audio_base64 = base64.b64encode(wav_file.getvalue()).decode('utf-8')
            
        return {"generated_audio_base64":audio_base64}

    def finalize(self):
        self.llm = None
        self.tts = None
