file_en = r"D:\data\cmn-en.txt\Tatoeba.cmn-en.en"
file_zh = r"D:\data\cmn-en.txt\Tatoeba.cmn-en.cmn"
output_file = 'cmn_large.txt'

with open(file_en, 'r', encoding='utf-8') as f_en, \
     open(file_zh, 'r', encoding='utf-8') as f_zh, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    count = 0
    for en_line, zh_line in zip(f_en, f_zh):
        en_text = en_line.strip()
        zh_text = zh_line.strip()
        
        # 过滤掉空行
        if not en_text or not zh_text:
            continue
            
        f_out.write(f"{en_text}\t{zh_text}\n")
        count += 1

print(f"合并完成！共生成 {count} 行有效的双语数据，保存为 {output_file}。")