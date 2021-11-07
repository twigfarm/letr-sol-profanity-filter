from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
import numpy as np

category_map = {
    "0": "일반글",
    "1": "공격발언",
    "2": "혐오발언"
}

app = Flask(__name__)

model = BertForSequenceClassification.from_pretrained('dobbytk/letr-sol-profanity-filter')
tokenizer = BertTokenizer.from_pretrained('dobbytk/letr-sol-profanity-filter')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print('Model load Finished!')

# 입력 데이터 변환
def convert_input_data(sentences):
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    MAX_LEN = 128
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks


def test_sentences(sentences):
    model.eval()
    inputs, masks = convert_input_data(sentences)
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
            
    with torch.no_grad():     
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    logits = outputs[0]
    logits = np.array(F.softmax(logits.detach().cpu()))
    category = np.argmax(logits)
    return {
        'Default': format(logits[0][0], ".4f"),
        'Offensive': format(logits[0][1], ".4f"),
        'Hate': format(logits[0][2],".4f"),
        'Category': category_map[str(category)]
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    predict = test_sentences([input_text])
    print(predict)
    return render_template('index.html', inputText=input_text, prediction=predict)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)