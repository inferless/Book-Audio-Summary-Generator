from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pypdf import PdfReader
from TTS.api import TTS
import torch
import io
import base64
import requests
import os
os.environ["COQUI_TOS_AGREED"]="1"

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
        self.prompts = prompts = [
            "Summarizing the following text from the chapters of a book.",
            "Write comprehensive notes summarizing the following text from the book.",
            """Based on the summaries provided, compose a comprehensive summary of the book titled '<Book-Name>' suitable for a speech. Follow this structure:**
                 1. **Summary of the Book '<Book-Name>':** Provide a brief overview capturing the essence of the book.
                 2. **Introduction:** Introduce the main themes, purposes, and significance of the book.
                 3. **Chapter Summaries:** Briefly summarize each chapter, highlighting key events, developments, and insights.
                 4. **Conclusion:** Conclude by summarizing the overall impact of the book and its contributions.
         
             **Ensure the speech is engaging, coherent, and maintains a consistent tone throughout."""
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
        
    def recursive_summarize(self,text_chunks, prompts_idx, max_tokens, batch_size):
        summaries = []
        for i in range(0, len(text_chunks), batch_size): 
            batch = "\n\n".join(text_chunks[i:i + batch_size])
            batch_summary = self.generate_summary(prompts_idx, max_tokens, batch)
            summaries.append(batch_summary)
            
        if len("\n\n".join(summaries).split(" "))>4000:
            return self.recursive_summarize(summaries, prompts_idx, max_tokens, batch_size)
        else:
            final_summaries = "\n\n".join(summaries)
            final_summary = self.generate_summary(2, max_tokens,final_summaries)
            return final_summary
        
    def infer(self,inputs):
        book_url = inputs['book_url']
        book_name = self.download_book(book_url)
        parsed_text = self.pdf_to_text(book_name)
        
        initial_summaries = [
            self.generate_summary(0, 200, chunk)
            for chunk in self.split_text_into_chunks(parsed_text, chunk_size=4000)
        ]
        final_summary = self.recursive_summarize(initial_summaries, 1, 1024, batch_size=5)
        
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
