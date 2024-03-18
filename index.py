from flask import Flask, render_template, request
import torch
from transformers import GPT2Tokenizer
import pickle

app = Flask(__name__)

# 모델 로드
with open('team3_blend5_try2.pkl', 'rb') as f:
    model = pickle.load(f)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # 웹페이지에서 입력받은 텍스트를 가져옴
    prompt_text = request.form['prompt_text']
    
    # 모델 입력 전처리
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
    input_ids = input_ids.to(device)
    
    # 모델 호출하여 예측 수행
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    
    # 모델 출력 후처리
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return render_template('result.html', input_text=prompt_text, generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
