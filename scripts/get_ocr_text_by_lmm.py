import datetime
import os
import fitz
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import trange

# 生成img
def pyMuPDF_fitz(pdfPath, imagePath,rotate,dpi_wanted):
    pdfDoc = fitz.open(pdfPath)
    num_pages = pdfDoc.page_count
    for pg in trange(pdfDoc.page_count):
        page = pdfDoc[pg]
        rotate = int(0) # 旋转角度
        dpi_wanted = 300     # 清晰度 
        zoom_factor = dpi_wanted / 72                   # PDF默认72dpi做基准 
        mat = fitz.Matrix(zoom_factor , zoom_factor).prerotate(rotate)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        if not os.path.exists(imagePath):  
            os.makedirs(imagePath)  
        out_file = os.path.join(imagePath  , f'images_{pg+1}.png')
        pix.save(out_file)
    return num_pages


# 获取文字
def get_text(model_path,num_pages,imagePath,prompt,max_new_tokens):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    results = []
    for i in trange(num_pages):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(imagePath  , f'images_{i+1}.png'),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        results.append(output_text[0])
    return results



if __name__ == "__main__":
    # PDF地址
    pdfPath = '31-53-00_起飞警告.pdf'
    # 需要储存图片的目录
    imagePath = './imgs'
    # pdf的页数
    num_pages = pyMuPDF_fitz(pdfPath, imagePath,0,300)
    # LLM
    model_path = "Qwen2.5-VL-7B-Instruct/Qwen/Qwen2___5-VL-7B-Instruct"
    prompt = "识别抽取图片中的文字，整理成连续通顺的文本"
    max_new_tokens  = 512
    results = get_text(model_path,num_pages,imagePath,prompt,max_new_tokens)
    with open("results.txt","w",encoding = "utf-8") as f:
        for i in results:
            f.write(i.strip())
            f.write('\n')