from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
from torch.nn import functional as F
from tqdm import tqdm

class Perplexity:
    
    def __init__(self,model,data):
        """
        init data and model with AutoTokenizer by transformers model
        """
        
        self.data = data
        
        # self.config = AutoConfig.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def cal_perplexity(self):
        """
        Calculate model's perplexity
        """

        self.encodings = self.tokenizer((self.data), return_tensors="pt")

        self.max_length = self.model.config.n_positions
        self.stride = 512
        self.seq_len = self.encodings.input_ids.size(1)

        self.nlls = []
        self.prev_end_loc = 0

        for self.begin_loc in tqdm(range(0, self.seq_len, self.stride)):
            self.end_loc = min(self.begin_loc + self.max_length, self.seq_len)
            self.trg_len = self.end_loc - self.prev_end_loc  # may be different from stride on last loop
            self.input_ids = self.encodings.input_ids[:, self.begin_loc:self.end_loc]
            self.target_ids = self.input_ids.clone()
            self.target_ids[:, :-self.trg_len] = -100

            with torch.no_grad():
                self.outputs = self.model(self.input_ids, labels=self.target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                self.neg_log_likelihood = self.outputs.loss

            self.nlls.append(self.neg_log_likelihood)

            self.prev_end_loc = self.end_loc
            if self.end_loc == self.seq_len:
                break

        self.ppl = torch.exp(torch.stack(self.nlls).mean())

        return self.ppl

    # def cal_perplexity(self):
    #     """
    #     Calculate model's perplexity
    #     """

    #     self.inputs = self.tokenizer(self.data, return_tensors = "pt")
    #     self.loss = self.model(input_ids = self.inputs["input_ids"], labels = self.inputs["input_ids"]).loss
    #     self.ppl = torch.exp(self.loss)

    #     return self.ppl

if __name__ == "__main__":

    model = 'gpt2-large'
    text = 'พระราชดำรัสที่จะให้มีชั้นเรียนขนาดใหญ่ให้หนุ่มสยามได้ศึกษาภาษาอังกฤษอย่างดี และจะทรงโปรดให้มีโรงเรียนมัธยมขึ้นในบางกอก ที่สอนทั้งภาษาอังกฤษและวิทยาศาสตร์จากตะวันตกด้วย พระบาทสมเด็จพระจุลจอมเกล้าเจ้าอยู่หัว พระราชทานที่ดินแก่พระเจ้าบรมวงศ์เธอ พระองค์เจ้าโสมาวดี ศรีรัตนราชธิดา กรมหลวงสมรรัตนสิริเชษฐ ตำบล ราชบูรณะ เมื่อวันที่ ๖ ธันวาคม ร.ศ. ๑๒๓ ตรงกับ พ.ศ. ๒๔๔๗ ที่ดินเลขที่ ๑๐๘ โฉนดเลขที่ ๑๒๘๖ พื้นที่ตามโฉนด ๒๕๖ ไร่ ๑ งาน ๙๖ ตารางวา'
    Calculate_Test = Perplexity(model,text)
    perplexity = Calculate_Test.cal_perplexity()
    
    print(f"Perplexity of {model} model is {perplexity}")