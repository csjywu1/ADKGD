# 读取旧的 reqirements.txt 文件
with open('reqirements.txt', 'r') as file:
    lines = file.readlines()

# 处理每一行，将空格替换为==
new_lines = []
for line in lines:
    # 分割每一行为包名和版本号，假设两者之间的空格数至少为2个
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        # 使用==连接包名和去除多余空白的版本号
        new_line = parts[0] + '==' + parts[1].strip()
        new_lines.append(new_line)
    else:
        # 如果行不含空格（即没有版本号），直接使用
        new_lines.append(line.strip())

# 将处理后的行写回新的 reqirements.txt 文件
with open('reqirements.txt', 'w') as file:
    for line in new_lines:
        file.write(line + '\n')
