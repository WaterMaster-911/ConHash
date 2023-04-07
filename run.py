import subprocess

# 定义要执行的脚本列表
scripts = ['Conformer.py']

# 打开日志文件以写入
# 循环执行脚本
for script in scripts:
    Log = 'RunLog'+"["+script+"]"
    with open(Log, 'w') as log_file:
        try:
            # 将输出重定向到日志文件
            subprocess.run(['python', script], check=True, stdout=log_file)
        except subprocess.CalledProcessError:
            log_file.write(f'{script} failed to run\n')
