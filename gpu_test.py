import torch as t

print("Cuda available?: ", t.cuda.is_available())
print(t.version.cuda)
print(t.cuda.get_arch_list())
print(t.cuda.is_available())
print(t.cuda.current_device())
print(t.cuda.device(0))
print(t.cuda.device_count())
print(t.cuda.get_device_name(0))
