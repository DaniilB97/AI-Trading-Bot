import time

print("start system scheck...")
time.sleep(1)

spinning_chars = ['/','-','\\','|',]

for i in range(1, 101):
    spinning_index = i % len(spinning_chars)
    spininning_chart = spinning_chars[spinning_index]
    user_output = f"Currecnt check status: {i}% [{spininning_chart}]"
    print(user_output, flush=True, end='\r')
    time.sleep(0.05)

print("System Checked!")