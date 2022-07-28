from tkinter import *
from tkinter.ttk import *

import json
import re

# custom module import
from generator import *

main_dir = "./"
output_file_path = main_dir+'save.json'
history = []

class MyFrame(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
 
        self.master = master
        self.master.title("korean ai novel generator")
        self.pack(fill=BOTH, expand=True)
 
        # result
        frame1 = Frame(self)
        frame1.pack(fill=X,expand=True)

        v=Scrollbar(self, orient='vertical')
        v.pack(side=RIGHT, fill='y')
        
        txtResult = Text(frame1, undo=True, yscrollcommand=v.set)
        txtResult.pack(fill=X, pady=10, padx=10, expand=True)
        
        if fileExist(main_dir, 'save.json'):
            first_result = str(loadOutput()['result'])
            txtResult.insert('end', first_result)

        frame2 = Frame(self)
        frame2.pack(fill=X)

        # progressbar
        frame3 = Frame(self)
        frame3.pack(fill=X,expand=True)

        pb = Progressbar(
            frame3,
            orient='horizontal',
            mode='indeterminate',
            length=500
        )
        # place the progressbar
        pb.grid(column=0, row=0, columnspan=2, padx=10, pady=20)
        
        def getTextList(text_box):
            txt = text_box.get("1.0", "end")
            txt_list = txt.split('\n')
            return txt_list
        
        def getLastLine(text_box):
            text_list = getTextList(text_box)

            while '' in text_list:
                text_list.remove('')

            return text_list[len(text_list)-1]
        
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

            regExp = re.compile(regTxt)
            gen_text = re.sub(regExp, addNewLine, generated_text)
            return gen_text

        #텍스트 생성 스크립트(generator.py) 실행
        def generate(event):
            pb.start(10)
            input_data = getLastLine(txtResult)
            output_data = novelMaker(input_data)
            
            result_data = txtResult.get("1.0", "end")
            result_data = result_data+beautifier(output_data)
            txtResult.delete("1.0","end")
            txtResult.insert("end", result_data)

            saveOutput(input_data, output_data, result_data)
            
            txtResult.see("end")
            pb.stop()

        # generate button
        btnEnter = Button(frame2, text="generate",command= lambda: generate(None))
        btnEnter.pack(side=RIGHT, padx=10, pady=10, expand=True)

        # debug
        frame4 = Frame(self)
        frame4.pack(fill=X,expand=True)

        btnInit = Button(frame4, text="init save.json",command= lambda: initOutput())
        btnInit.pack(side=RIGHT, padx=10, pady=10)

def main():
    root = Tk()
    root.geometry("600x550+100+100")
    app = MyFrame(root)
    root.mainloop()
 
 
if __name__ == '__main__':
    main()