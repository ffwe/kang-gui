import torch
import transformers
import json
import re
import glob
import sys

from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2Config

#상수 정의
pretrained_model_name="skt/kogpt2-base-v2"
main_dir = "./"
output_file_path = main_dir+'save.json'
checkpoint_file_path = main_dir+'model.pt'
gpu_activate = True

#옵션값 로드
def loadOption():
    with open(main_dir+'option.json', 'r') as json_file:
        option_data = json.load(json_file)
    return option_data

option = loadOption()
generator_option = option['generator_option']

# "./" 에 저장된 모델 불러오기
def loadModel(model_param):
  device = torch.device('cpu')
  fine_tuned_model_ckpt = torch.load(
      checkpoint_file_path,
      map_location=device,
  )
  model_param.load_state_dict(fine_tuned_model_ckpt)
  model_param.eval()
  return model_param

#파일 존재 확인
def fileExist(dir_path, file_name):
    file_list = glob.glob(f''+dir_path+file_name)
    return len(file_list) > 0

#텍스트 생성기 함수
def generator(prompt, model_param):
  prompt_ids = tokenizer.encode(prompt)

  if gpu_activate:
    inp = torch.tensor(prompt_ids)[None].cuda()
  else:
    inp = torch.tensor([prompt_ids])

  model = loadModel(model_param)
  preds = model.generate(inp,
                            max_length=generator_option['max_length'],
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            repetition_penalty=generator_option['repetition_penalty'],       
                            use_cache=True
                            ) 

  generated = tokenizer.decode(preds[0].cpu().numpy())
  return generated

#텍스트 문장마다 개행
def beautifier(generated_text):
    def addNewLine(m):
        return str(m.group())+'\n'
    endSpecials = ('.', '?', '!')
    endQuotes = ('','"',"'",'”','’')
    regTxt = ''
    for special in endSpecials:
        for quote in endQuotes:
            regTxt = regTxt+'(\\'+special+quote+'\s)'
            if not (special == endSpecials[-1] and quote == endQuotes[-1]):
                regTxt = regTxt+'|'   

#save.json 저장
def saveOutput(input_param,output_param,result_param):
    file_path = output_file_path
    obj = {"input":input_param,"output":output_param,"result":result_param}
    with open(file_path, 'w') as my_file:
        json.dump(obj, my_file, indent=4)

#save.json 초기화
def initOutput():
  file_path = output_file_path
  obj = {"input":'',"output":'',"result":''}
  with open(file_path, 'w') as my_file:
    json.dump(obj, my_file, indent=4)

#save.json 로드
def loadOutput():
    file_path = output_file_path
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)
        return json_data

#novelMaker
def novelMaker(input_text):
  input_data = input_text
  file_path = output_file_path

  if not fileExist(main_dir, 'save.json'):
    initOutput()

  if gpu_activate:
    text = generator(input_data, model.cuda())
  else:
    text = generator(input_data, model)

  output = text.replace(input_data,'',1)
  print("generated_text : "+text)
  print("output : "+output)
  return output

def testOutput():
  file_path = output_file_path
  with open(file_path, 'r') as json_file:
    json_data = json.load(json_file)
    print("input : "+json_data["input"])
    print("output : "+json_data["output"])
    print("result : "+json_data["result"])

#메인 로직

#pre-tunning 데이터 불러오기 (kogpt2)
tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name,
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>') 

model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)

checkpoint_is_null = fileExist(main_dir, 'model.pt')

if not checkpoint_is_null:
  torch.save(model.state_dict(), checkpoint_file_path)
else:
  model = loadModel(model)

#initOutput for test
#initOutput()

#첫번째 인자 넘겨받기
# input = ''
# arg_len = len(sys.argv)
# if arg_len > 1:
#   input = str(sys.argv[1])

# novelMaker(input)
# testOutput()