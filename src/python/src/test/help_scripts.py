from src.test import ECGDependentTest
from src.code.ECG_loader import PTB_ECG

import Tkinter as tk
import os
import logging
import matplotlib.pyplot as plt

def get_filename_without_extension(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class InvertedECGResolver(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.createForm()
        self._inverted_files_list = []
        directory = r'..\..\..\..\data\ptb_database_csv'
        # print os.getcwd()
        files_names = []
        with open(os.path.join(directory, 'info.txt'), 'r') as f:
            for line in f:
                files_names.append(line.strip())
        self.full_files_names = [os.path.join(directory, file_name) for file_name in files_names]
        self.i = 0
        self.show_file()

    def show_file(self):
        filename = self.full_files_names[self.i]
        ecg = PTB_ECG(filename)
        __, p = plt.subplots(2, sharex=True)
        p[0].plot(ecg.getLowFreq(), 'g-')
        p[0].set_title('Normal')
        p[1].plot(-ecg.getLowFreq(), 'g-')
        p[1].set_title('Inverted')

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

    def process_answer(self, inverted):
        logging.info('file %s is %s', self.full_files_names[self.i], 'inverted' if inverted else 'not inverted')
        if inverted:
            self._inverted_files_list.append(get_filename_without_extension(self.full_files_names[self.i]))
        logging.info('Inverted files: %s', str(self._inverted_files_list))

        self.i += 1
        plt.close()
        self.show_file()

    def createForm(self):
        self.invertedBtn = tk.Button(self)
        self.invertedBtn['text'] = 'inverted'
        self.invertedBtn.pack({"side": "left"})
        self.invertedBtn["command"] = lambda: self.process_answer(True)

        self.notInvertedBtn = tk.Button(self)
        self.notInvertedBtn['text'] = 'not inverted'
        self.notInvertedBtn.pack({"side": "left"})
        self.notInvertedBtn["command"] = lambda: self.process_answer(False)

root = tk.Tk()
root.wm_attributes("-topmost", 1)
app = InvertedECGResolver(master=root)
app.mainloop()
root.destroy()

