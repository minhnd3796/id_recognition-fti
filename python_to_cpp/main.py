import sys
from matplotlib import pyplot as plt
import Id
from recognise_id_number import crop_id, idToStr

id_only_img = crop_id(sys.argv[1])

id_str = idToStr(id_only_img, sys.argv[2], sys.argv[3])

# Get 4 vars
name = Id.get_name()
addr = Id.get_addr()
dob = Id.get_dob()

print("id:", id_str)
print("name:", name)
print("addr:", addr)
print("dob:", dob)

plt.imshow(id_only_img, cmap = 'gray')
plt.show()
