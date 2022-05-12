import imageio

gifs = []
step = 360//20
for frame_from in range(0, 360-step+1, step):
    gif = imageio.get_reader(f"../images/cut_from_{frame_from}.gif")
    gifs.append(gif)

print(f"total {len(gifs)} gifs")

merged = imageio.get_writer("../images/cut_merged.gif")

frame_number = 0
for gif in gifs:
    for _ in range(gif.get_length()):
        print(f"frame {frame_number}")
        image = gif.get_next_data()
        merged.append_data(image)
        frame_number += 1
    gif.close()

merged.close()
print("done")

