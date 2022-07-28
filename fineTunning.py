import torch
import transformers

import fastai
import re
import json
import sys

from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2Config
from fastai.text.all import *

#상수 정의
main_dir = './'
pretrained_model_name='skt/kogpt2-base-v2'
checkpoint_file = main_dir+'model.pt'
gpu_activate = True

#옵션값 로드
def loadOption():
    with open(main_dir+'option.json', 'r') as json_file:
        option_data = json.load(json_file)
    return option_data

option = loadOption()
fineTunning_option = option['fineTunning_option']

#model 불러오는 함수
def loadModel(model_param):
  device = torch.device('cpu')
  fine_tuned_model_ckpt = torch.load(
      checkpoint_file,
      map_location=device,
  )
  model_param.load_state_dict(fine_tuned_model_ckpt)
  return model_param

#model input output tokenizer
class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x): 
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))

#gpt2 ouput is tuple, we need just one val
class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]

#파인튜닝 반복문
def fineTunningLoop(raw_dir,model_param,start,end):
  #model load
  model = loadModel(model_param)

  files = glob.glob(f''+raw_dir+'/*.txt')
  i = 0
  max = end
  if end > len(files):
    max = len(files)

  print("max : "+str(max))

  for file in files:
    if i >= start:
      print('['+str(i+1)+'/'+str(len(files))+']'+file)
      
      with open(file,encoding="UTF-8") as f:
        lines = f.read()
      lines=" ".join(lines.split())
        
    # 튜닝 전에 특정 단어 삭제 (ex) ".=." -> "==== 1화 ====" 등 삭제.)
      ban_cards = fineTunning_option["ban_cards"]
      for card in ban_cards:
        lines=re.sub(card, '', lines)
      print('줄 수 : '+str(len(lines)))

      #split data
      train=lines[:int(len(lines)*0.9)]
      test=lines[int(len(lines)*0.9):]
      splits = [[0],[1]]

      #init dataloader
      tls = TfmdLists([train,test], TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
      batch,seq_len = 8,256
      dls = tls.dataloaders(bs=batch, seq_len=seq_len)
      # dls.show_batch(max_n=2)
      
      learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()
      lr=learn.lr_find()
      print(lr)
      learn.fit_one_cycle(5, lr)
      # learn.fine_tune(3)

    i = i + 1

    if i > max-1:
      #save to model.pt
      torch.save(learn.model.state_dict(), checkpoint_file)
      print("saved "+checkpoint_file)
      model = learn.model
      break

  return model

#텍스트 생성 로직
def generator(prompt, model_param):
  prompt_ids = tokenizer.encode(prompt)

  if gpu_activate:
    inp = torch.tensor(prompt_ids)[None].cuda()
  else:
    inp = torch.tensor([prompt_ids])

  preds = model_param.generate(inp,
                            max_length=50,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            repetition_penalty=10.0,       
                            use_cache=True
                            ) 

  generated = tokenizer.decode(preds[0].cpu().numpy())
  return generated

#텍스트 문장마다 개행
def beautifier(generated_text):
  regExp = re.compile('(\." )|(\.\')|(\.\s)|(\?)|(\!)')
  text_list = re.split(regExp,generated_text)
  gen_text = ''

  temp = ''
  for chunk in text_list:
    if chunk is None or chunk == ' ' or chunk == '':
      continue
    elif re.match(regExp,chunk):
      temp = temp+chunk
      gen_text = gen_text + temp + '\n'
      temp = ''
    else:
      temp = temp+chunk
  return gen_text


#download model and tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name,
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>') 
model = AutoModelWithLMHead.from_pretrained(pretrained_model_name)

checkpoint = glob.glob(f''+checkpoint_file)
checkpoint_is_null = len(checkpoint) == 0

if checkpoint_is_null:
  torch.save(model.state_dict(), checkpoint_file)
  print("model.pt 파일이 없어서 새로 생성.")
else:
  model = loadModel(model)
  print("이미 존재하는 model.pt 파일을 불러옴.")

start = 0
end = len(glob.glob(f''+main_dir+'/raw'+'/*.txt'))

arg_len = len(sys.argv)
if arg_len > 1:
   start = int(sys.argv[1])

if arg_len > 2:
    end = int(sys.argv[2])

model = fineTunningLoop(main_dir+'/raw',model,start,end)