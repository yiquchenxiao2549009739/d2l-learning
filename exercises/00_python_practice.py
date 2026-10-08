# 第 00 课练习：请自己替换下面两处 None。
# None 暂时代表“尚未填写”。不要改检查部分。

x = 3
w = 2
b = 1

# TODO 1：写出计算 y 的公式，不要直接填写计算好的数字。
y = None

# TODO 2：计算输入为 5 时的预测，也使用 w 和 b。
new_x = 5
new_y = None

# 以下是练习检查，暂时不用读懂。
if y is None or new_y is None:
    print("还未完成：请替换两处 None，保存后重新运行。")
elif y == w * x + b and new_y == w * new_x + b:
    print("计算正确！请在聊天里解释你写的两条公式。")
    print("第一个预测：", y)
    print("第二个预测：", new_y)
else:
    print("结果还不正确。先检查乘法符号和用到的输入变量。")
    print("你的结果：", y, new_y)
